"""Time the baseline on a generated room and corridors; no arena data."""
import time

from equipo_jar.cobertura_plan import Grid, Pose, Settings, make_plan


def main():
    size, resolution = 120, 0.05
    for divided in (False, True):
        data = []
        for j in range(size):
            for i in range(size):
                wall = i in (0, size - 1) or j in (0, size - 1)
                if divided and i % 24 == 0 and i and j < size - 15:
                    wall = True
                data.append(100 if wall else 0)
        grid = Grid(size, size, resolution, data)
        before = time.perf_counter()
        plan = make_plan(grid, Pose(0.75, 0.75, 0.0), Settings())
        elapsed = time.perf_counter() - before
        print(f'corridors={divided}: {elapsed:.2f}s, {len(plan.poses)} views, '
              f'predicted={plan.predicted_count}/{plan.target_count}, '
              f'candidates={plan.candidate_count}', flush=True)


if __name__ == '__main__':
    main()
