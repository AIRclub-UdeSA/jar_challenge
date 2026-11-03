"""Local CPU/compatibility evidence; no ROS, navigation or score claims.

PYTHONPATH=src/equipo_jar python3 pruebas/medir_repaso.py --output build/.../cpu.json
Optional --history-module loads the reviewed pure history file from PR #58.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

from equipo_jar.cobertura_plan import Camera, Settings
from equipo_jar.cobertura_repaso import propose_extra_views, pending_samples, visibility_grid
from equipo_jar.mapa import Grid, Pose
from test_repaso_cobertura import room, outside_history, Seen, sees


def resample(grid, resolution, origin, width, height):
    values = []
    for j in range(height):
        for i in range(width):
            x,y = (i+0.5)*resolution,(j+0.5)*resolution
            wx = origin.x+math.cos(origin.yaw)*x-math.sin(origin.yaw)*y
            wy = origin.y+math.sin(origin.yaw)*x+math.cos(origin.yaw)*y
            cell = grid.cell(wx,wy)
            values.append(grid.data[cell[1]*grid.width+cell[0]] if cell is not None else -1)
    return Grid(width,height,resolution,values,origin)


def run(output, history_module):
    static,live = room()
    history = outside_history()
    dense = tuple(Seen(Pose(v.pose.x+(k+1)*1e-7,v.pose.y,v.pose.yaw),float(k))
                  for k in range(1200) for v in (history[k%len(history)],))
    closed_static,closed_live = room(closed=True)
    cases = (
        ('room',static,live,history),
        ('no_history',static,live,()),
        ('1200_distinct_poses',static,live,dense),
        ('live_3cm',static,resample(live,0.03,Pose(0,0,0),200,134),history),
        ('live_shifted_rotated',static,resample(live,0.05,Pose(-0.1,-0.2,0.03),126,88),history),
        ('closed_room',closed_static,closed_live,history),
    )
    results = []
    for name,a,b,observed in cases:
        times=[]
        for _ in range(3):
            # Fresh grids: do not hide per-call mask construction in warm caches.
            a=Grid(a.width,a.height,a.resolution,a.data,a.origin)
            b=Grid(b.width,b.height,b.resolution,b.data,b.origin)
            begin=time.perf_counter()
            result=propose_extra_views(a,b,Pose(1,2,0),Pose(1,2,0),observed,(),Camera(),
                                       Settings(),limit=6,failed_shift_m=0.4)
            times.append(time.perf_counter()-begin)
        item=dict(name=name,seconds=times,median_s=statistics.median(times),
                  pending_count=result.pending_count,proposals=len(result.views))
        results.append(item)
        print(json.dumps(item),flush=True)
    visual=visibility_grid(static,live)
    pending=pending_samples(static,live,history,Camera(),Settings())
    shadow=[p for p in pending if 3.7<p[0]<4.5 and 1.8<p[1]<2.2]
    robot,start=Pose(1,2,0),Pose(1,2,0)
    trace=[]
    for _ in range(6):
        result=propose_extra_views(static,live,robot,start,history,(),Camera(),Settings(),
                                   limit=6,failed_shift_m=0.4)
        if not result.views:
            break
        robot=result.views[0].pose
        recovered=sum(sees(visual,robot,p) for p in shadow)
        trace.append(dict(pose=asdict(robot),gain=result.views[0].gain,
                          pending=result.pending_count,shadow_samples_visible=recovered))
        if recovered:
            break
        history+=(Seen(robot,100.0+len(trace)),)
    compatibility=None
    if history_module:
        spec=importlib.util.spec_from_file_location('reviewed_mission_history',history_module)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        real_history=tuple(module.ObservedView(v.pose,float(k))
                           for k,v in enumerate(outside_history()))
        attempt=module.AttemptedView(Pose(2,2,0),'failed')
        original=propose_extra_views(static,live,Pose(1,2,0),start,outside_history(),
                                     (attempt,),Camera(),Settings(),limit=6,failed_shift_m=0.4)
        actual=propose_extra_views(static,live,Pose(1,2,0),start,real_history,
                                   (attempt,),Camera(),Settings(),limit=6,failed_shift_m=0.4)
        assert actual==original
        compatibility=dict(source_sha256=hashlib.sha256(Path(history_module).read_bytes()).hexdigest(),
                           same_result=True)
    root=Path(__file__).resolve().parents[1]
    source=root/'src/equipo_jar/equipo_jar/cobertura_repaso.py'
    evidence=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  measurements=results,closed_loop=trace,mission_history=compatibility,
                  limitations='Generated geometry, static occluder, ideal reached poses; no detector or navigation.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--history-module')
    args=parser.parse_args()
    run(args.output,args.history_module)
