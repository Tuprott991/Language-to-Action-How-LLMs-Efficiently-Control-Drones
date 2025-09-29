from data_generate.utils import reconstruct_path
import heapq

def rrt_stub(grid, start, goal):
    """ Fake RRT* path: random walk towards goal in 3D """
    path = [start]
    current = start
    size, _, altitude_levels = grid.shape
    while current != goal:
        dx = 1 if goal[0] > current[0] else -1 if goal[0] < current[0] else 0
        dy = 1 if goal[1] > current[1] else -1 if goal[1] < current[1] else 0
        dz = 1 if goal[2] > current[2] else -1 if goal[2] < current[2] else 0
        next_pos = (current[0]+dx, current[1]+dy, current[2]+dz)
        if (0 <= next_pos[0] < size and 0 <= next_pos[1] < size and 0 <= next_pos[2] < altitude_levels and grid[next_pos] == 0):
            path.append(next_pos)
            current = next_pos
        else:
            break
    return path if path[-1] == goal else None