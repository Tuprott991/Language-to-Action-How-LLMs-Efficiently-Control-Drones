import heapq
from data_generate.utils import reconstruct_path

def heuristic(a, b):
    # Manhattan distance in 3D
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def astar(grid, start, goal):
    size, _, altitude_levels = grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    directions = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for dx, dy, dz in directions:
            neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
            if (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and 0 <= neighbor[2] < altitude_levels and grid[neighbor] == 0):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None
