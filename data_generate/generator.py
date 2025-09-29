from data_generate.utils import generate_grid, sample_free_cell, path_to_minispec
from data_generate.dijkstra import dijkstra
from data_generate.rrt import rrt_stub
from data_generate.astar import astar
import json
import random
from data_generate.advanced_template import TEMPLATES

def generate_dataset(n=100, size=10, out_file="drone_dataset.jsonl"):
    from data_generate.utils import abstract_map_description
    altitude_levels = 3
    with open(out_file, "w") as f:
        for _ in range(n):
            grid, obstacles, nofly_zones = generate_grid(size=size, 
                                                         altitude_levels=altitude_levels,
                                                         obstacle_ratio=0.05,
                                                         nofly_ratio=0.05)
            start = sample_free_cell(grid)
            goal = sample_free_cell(grid)
            while goal == start:
                goal = sample_free_cell(grid)

            planner = random.choice(["A*", "Dijkstra", "RRT*"])
            if planner == "A*":
                path = astar(grid, start, goal)
            elif planner == "Dijkstra":
                path = dijkstra(grid, start, goal)
            else:
                path = rrt_stub(grid, start, goal)

            if path is None:
                continue

            # Random altitude per mission
            altitude = round(random.uniform(1.0, 5.0), 2)
            failure = random.random() < 0.2
            hierarchy = random.random() < 0.5
            desc = random.choice(TEMPLATES).format(s=start, g=goal)
            plan = path_to_minispec(path, altitude=altitude, failure=failure, hierarchy=hierarchy)
            # Only show a random subset of obstacles to the LLM (e.g., 50% visible)
            obstacle_visible_ratio = 0.5 # Means 50% of obstacles are visible in the data
            # Only obstacles for abstract_map
            def obstacle_map_desc(obstacles, ratio):
                num_visible = int(len(obstacles) * ratio)
                visible_obstacles = random.sample(obstacles, num_visible) if obstacles and num_visible > 0 else []
                return "; ".join([f"Obstacle at ({x},{y},{z})" for (x, y, z) in visible_obstacles])
            abstract_map = obstacle_map_desc(obstacles, obstacle_visible_ratio)
            # Only no-fly zones for noflyzone
            noflyzone_desc = "; ".join([f"No-fly zone at ({x},{y}) (all altitudes)" for (x, y) in nofly_zones])

            entry = {
                "instruction": desc,
                "planner": planner,
                "output": plan,
                "grid_shape": grid.shape,
                "start": start,
                "goal": goal,
                "altitude": altitude,
                "failure_case": failure,
                "hierarchical": hierarchy,
                "abstract_map": abstract_map,
                "noflyzone": noflyzone_desc
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    generate_dataset(30000, size=10)