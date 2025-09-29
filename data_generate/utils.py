import numpy as np
import random

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def generate_grid(size=10, altitude_levels=3, obstacle_ratio=0.1, nofly_ratio=0.1):
    # 3D grid: (x, y, z)
    grid = np.zeros((size, size, altitude_levels), dtype=int)
    obstacles = []
    nofly_zones = []
    # Place obstacles as cubes
    num_obstacles = int(obstacle_ratio * size * size * altitude_levels)
    for _ in range(num_obstacles):
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        z = random.randint(0, altitude_levels-1)
        grid[x, y, z] = 1
        obstacles.append((x, y, z))
    # Place no-fly zones as 2D regions (all altitudes)
    num_nofly = int(nofly_ratio * size * size)
    for _ in range(num_nofly):
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        grid[x, y, :] = 2
        nofly_zones.append((x, y))
    return grid, obstacles, nofly_zones

def sample_free_cell(grid):
    size, _, altitude_levels = grid.shape
    while True:
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        z = random.randint(0, altitude_levels-1)
        if grid[x, y, z] == 0:
            return (x, y, z)
        
def path_to_minispec(path, altitude=1.0, failure=False, hierarchy=False):
    spec = f"takeoff({altitude}); "
    if hierarchy and len(path) > 2:
        mid = len(path)//2
        spec += "if True { "
        for (x, y, z) in path[:mid]:
            spec += f"goto({x},{y},{z}); "
        spec += "} "
        spec += "[2] { log('scanning'); delay(1); } "
        for (x, y, z) in path[mid:]:
            spec += f"goto({x},{y},{z}); "
    else:
        for (x, y, z) in path:
            spec += f"goto({x},{y},{z}); "
    if failure:
        spec += "re_plan(); "
    spec += "land();"
    return spec

# Abstract map description generator
def abstract_map_description(obstacles, nofly_zones, obstacle_visible_ratio=1.0):
    desc = []
    # Randomly select a subset of obstacles to show
    num_visible = int(len(obstacles) * obstacle_visible_ratio)
    visible_obstacles = random.sample(obstacles, num_visible) if obstacles and num_visible > 0 else []
    for (x, y, z) in visible_obstacles:
        desc.append(f"Obstacle at ({x},{y},{z})")
    for (x, y) in nofly_zones:
        desc.append(f"No-fly zone at ({x},{y}) (all altitudes)")
    return "; ".join(desc)