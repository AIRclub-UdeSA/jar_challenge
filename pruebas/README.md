# Coverage checks

Current command-position/time proxies, explicit goal-result records and noise
sensitivity tools are documented in
[`EXECUTION_AUDIT.md`](../notes/marcorita/EXECUTION_AUDIT.md).
Historical identity-only metrics remain for comparisons; they do not model
mission's in-place command substitution, dynamic boxes or actual camera frames.

From the repository root, in Ubuntu 22.04 with ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$PWD/src/equipo_jar${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest pruebas/test_cobertura_plan.py pruebas/test_cobertura_node.py -q
python3 pruebas/benchmark_cobertura.py
bash tools/verificar.sh --template
```

The first suite checks generated geometry. The second needs ROS, but no
simulator; it checks waiting for map/TF, fresh localization, late subscribers,
parameter validation, pose corrections, cancellation and live replanning.
`src/equipo_jar/test/test_cobertura.py` runs these suites from `colcon test`,
so the normal verifier checks coverage too. Its wrapper contains no test geometry.
The benchmark measures only the
planner, on generated grids; it is not a navigation or detection benchmark.
Generated maps and test coordinates stay outside `src/`.

## Compare plans on all six practice maps

With the simulator installed and sourced:

```bash
python3 pruebas/evaluar_cobertura.py \
  --simulator "$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)" \
  --output build/verificacion/cobertura/evaluacion-final
```

This reads maps from the simulator install directory. It writes JSON, CSV,
a Markdown comparison and plots under ignored `build/`; it does not copy maps
into source. It evaluates three starts per map and compares the initial rule
against fine candidate recovery plus one bounded 2-opt sweep.

The improved plan retains every initial observation, appends useful views
within the configured budgets, then reorders them only when both estimated
total time and mean first-view time improve. It never drops a view just because
its sampled mask looks redundant: unsampled cells could lose visibility.

Plots show predicted full-grid visibility, sampled residuals, shortest grid
legs and the estimated coverage curve. Residuals distinguish targets outside
the sampled candidate model from targets visible but deferred by the goal
budget. The report also counts targets disconnected before footprint inflation.
Some remaining gaps can come from the conservative circular footprint, the
map itself or the discrete heading model; the report does not call them all
physically impossible.

Runtime measurements are one run per variant and case, alternating variant
order. They include diagnostic path reconstruction. Travel estimates omit
acceleration, localization delays, new obstacles and detector workload.
Return distance is separate. These are planning proxies, not a run-time promise.

## Inspect a plan in simulation

Start the official simulator and the practice judge using the team's simulation
guide. Use `pruebas.launch.py` while localization is missing, so `map → odom`
exists. That identity transform is only a test aid, not localization.

After building and sourcing `install/setup.bash`, in another terminal:

```bash
python3 pruebas/capturar_plan_cobertura.py
```

It subscribes to the map and plan and writes `build/verificacion/cobertura/plan.png`
and a JSON of the observation poses. Plotting uses matplotlib and NumPy, only
for this development tool; the shipped coverage node uses neither.

The numbered arrows are observation goals, not the path Nav2 will drive. Actual
visibility, camera processing, goal success and failed-goal recovery still need
integration tests with mission, navigation and perception.

When stopping a launch from a Python probe, send SIGINT to the launch parent.
ROS launch forwards it to its children. Sending it to the whole process group
first interrupts children twice and can break their cleanup. Use the known
process group only to force-stop your own probe after a shutdown timeout.

## Evaluate time budgets and mission selection

With the source Python path and simulator sourced as above:

```bash
python3 pruebas/evaluar_presupuesto.py \
  --simulator "$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)" \
  --output build/verificacion/cobertura/presupuesto
```

This evaluates all six practice maps at three starts. JSON, CSV and Markdown
include intended/mission-selected goals, dense visibility at 60/120/300/480 s,
raw-connected denominators, startup wait and estimated travel/turn/dwell.
The actual mission selector is imported from `exploracion.py`.
The model assumes every goal succeeds; observations and inspections are not
measured. A view earns predicted visibility after its estimated dwell completes.

The nominal start on each map gets an exhaustive missing-target witness search.
It separates raw disconnection, missing sampled positions and limits of the
current reachable-cell-center/camera model. A second search checks safe centers
in other components. A no-witness result is not physical impossibility.
Use `--diagnose none` for a shorter run, `--starts nominal --maps laberinto_1`
for a pilot, and `--profile` for one separately instrumented planner execution.
All runtime outputs must stay under this checkout's ignored `build/` directory.

For paired runtime checks, save the old pure planner revision under `build/`
and run:

```bash
python3 pruebas/comparar_rendimiento.py \
  --reference build/verificacion/cobertura/referencia/cobertura_plan.py \
  --simulator "$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)"
```

The reference file is Python code and must be a trusted repository revision.
The comparison does one warmup per variant/map and three alternating paired
repetitions. Every execution asserts identical poses, target counts and route
metrics. Run without another CPU-heavy evaluation; results are machine-specific.
Timings exclude diagnostic reconstruction and the exhaustive oracle.

## Verify calculation reuse

For calculation-only optimizations, use the paired timing command above
without `--allow-reordering`: exact route and metric parity is required.
Free-cell masks are cached per immutable Grid and occupancy threshold; the
latest threshold replaces the cache. Map updates create a new Grid. The ROS
suite checks that changing the threshold live creates a new usable plan.
Generated geometry/cache checks can also run without ROS:

```bash
python3 -m pytest pruebas/test_calculo_geometrico.py -q
```

## Compare fixed-view observation ordering

The runtime keeps the existing selected view set. After 2-opt, two bounded
relocation sweeps can move a view earlier or later. The cleanup sweep revisits
choices that later moves can unlock. Each accepts only improvements
in sampled mean first-view time (or equal mean with shorter search), with no
increase in estimated search or search-plus-return. Sampled coverage must not
decrease on the configured predicted-clock grid.

`order_check_interval_s` is the check spacing; `order_start_delay_s` estimates
the initial wait before the route. Both apply live. The estimate does not read
or control mission's clock. Recheck it when startup wait or actual planning/TF
delays change. Coverage between samples/checks and physical detection remain
unverified; the full-grid offline report is still needed.

Use the budget evaluator above for the current planner. To compare isolated
ordering policies against an earlier evaluator's saved poses:

```bash
python3 pruebas/experimentar_orden.py \
  --reference build/verificacion/cobertura/presupuesto-optimizado/budget.json \
  --simulator "$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)" \
  --modes sample-phase60 --output build/verificacion/cobertura/orden-experimento
```

The script does not regenerate the reference plan. It evaluates retained views
with full-grid visibility and compares experimental relocation rules. The
`sample-phase60` policy uses minute checks aligned with the saved startup delay.
Other policies can regress early coverage and are not runtime defaults.

For paired complete-planner CPU measurements when only order changes, use
`comparar_rendimiento.py --allow-reordering` with a trusted saved planner file.
It checks identical view sets/target counts across versions and deterministic
order/metrics within each version. Without the flag, exact route/metric parity
is still required. Estimated travel savings are separate from CPU time.

## Observation clearance

Transit still uses `robot_radius + safety_margin`. Observation centers have a
separate minimum (`view_clearance`, 0.24 m) and preferred stand-off
(`view_preferred_clearance`, 0.30 m). Coarse sampling uses preferred centers;
fine recovery exhausts those before adding a tighter witness for a missed
target. Every point, including an optional start view, obeys the minimum.
Neither setting reduces transit clearance. Both apply live.

The current padded Nav2 rear corner is about 0.2341 m from base_footprint.
Recheck the minimum when chassis dimensions or footprint padding change.
This nominal-map rule does not certify arrival error, localization, actual
walls or unmapped boxes. Navigation remains responsible for each movement.
Separate observation clearance avoids closing transit passages just to keep
stops farther from walls, although the existing circular transit model still
has its own reachability limits.

```bash
python3 -m pytest pruebas/test_espacio_observacion.py -q
```

The budget evaluator uses eligible minimum-clearance observation centers in
its missing-view oracle. Compare final dense visibility and early budgets:
new viewpoints can lose visibility or add travel. The exact-view CPU/order
comparison is not an acceptance test for changed geometry. For a controlled
baseline replay only, set both observation clearances to zero before /map;
that restores the previous transit-only candidate policy.

## Compare repeated-position grouping

Whole-position and pair grouping remain offline experiments. They retain every
saved pose/heading and compare periodic/strict sampled guards; full-grid budgets
are checked separately. Current prototypes offer too little benefit to add to
runtime. To reproduce against saved routes:

```bash
python3 pruebas/experimentar_agrupacion.py \
  --reference build/verificacion/cobertura/espacio-orden-integrado/budget.json \
  --simulator "$(ros2 pkg prefix --share yahboom_rosmaster_gazebo)" \
  --modes pairs-time pairs-mean pairs-dominance pairs-clock \
  --output build/verificacion/cobertura/agrupacion-experimento
```

## Current mission integration

Coverage publishes an initial route. Mission in current development remembers
finished points, including failed goals, and selects the next pending position
and heading. Coverage does not remove those points or track actual observation.
See `notes/marcorita/INTEGRATION.md` for execution tolerances, deadlines and the
remaining perception/navigation agreements.

For a handoff check, use the official simulator and practice judge following
`docs/team/SIMULATION.md`. Match the ROS domain in each terminal.

An older communication-only fixture used fake navigation before Nav2 was
integrated. Do not start that fake action server alongside current Nav2.
That probe exposed the map about 54 s before robot TF and ended by inactivity.
Keep the failed attempt as evidence. Preparing sensors before a geometry
comparison isolates startup delays; it does not validate production startup.

```bash
ros2 launch equipo_jar competencia.launch.py use_sim_time:=true victimas:=2
python3 pruebas/capturar_plan_cobertura.py
```

Start the simulator and judge separately following the team guide. Pass the
actual practice world's count instead of 2. The launch now includes real AMCL
and Nav2. A captured plan alone proves neither travel nor processed images.
