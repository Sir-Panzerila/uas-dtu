import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

# 4-Directional movements: Up, Down, Left, Right (Diagonal not allowed)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def manhattan_distance(p1, p2):
    """Calculates Manhattan distance |x1 - x2| + |y1 - y2|"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def parse_grid(grid_str_list):
    """Extracts Start, Goal, and dimensions from character grid"""
    start = None
    goal = None
    rows = len(grid_str_list)
    cols = len(grid_str_list[0])
    for r in range(rows):
        for c in range(cols):
            val = grid_str_list[r][c]
            if val == 'S':
                start = (r, c)
            elif val == 'G':
                goal = (r, c)
    return start, goal, rows, cols

def a_star_search(grid):
    start, goal, rows, cols = parse_grid(grid)
    start_time = time.perf_counter()
    
    # Priority Queue stores: (f_score, current_node)
    open_heap = []
    heapq.heappush(open_heap, (0 + manhattan_distance(start, goal), start))
    
    # Cost from start to current node
    g_score = {start: 0}
    came_from = {}
    explored_nodes = set()
    
    path_found = False
    
    while open_heap:
        _, current = heapq.heappop(open_heap)
        explored_nodes.add(current)
        
        if current == goal:
            path_found = True
            break
            
        r, c = current
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            
            # Check grid bounds
            if 0 <= nr < rows and 0 <= nc < cols:
                # UAV cannot cross obstacles
                if grid[nr][nc] == '#':
                    continue
                    
                tentative_g = g_score[current] + 1  # Cost per move is 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + manhattan_distance(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))
                    
    exec_time = time.perf_counter() - start_time
    
    # Reconstruct optimal path if one exists
    path = []
    if path_found:
        curr = goal
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.append(start)
        path.reverse()
        total_cost = g_score[goal]
    else:
        total_cost = None
        
    return path_found, path, total_cost, len(explored_nodes), exec_time, explored_nodes, start, goal

def visualize_and_save(grid, path, explored, start, goal, filename, title):
    rows = len(grid)
    cols = len(grid[0])
    
    # Create an RGB image matrix
    # 0 = Free (White), 1 = Obstacle (Black), 2 = Explored (Light Blue), 3 = Path (Red)
    img = np.ones((rows, cols, 3))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '#':
                img[r, c] = [0.1, 0.1, 0.1]  # Dark grey / Obstacle
                
    for (r, c) in explored:
        if grid[r][c] != '#':
            img[r, c] = [0.75, 0.88, 1.0]  # Explored cells (light blue)
            
    for (r, c) in path:
        img[r, c] = [1.0, 0.3, 0.3]  # Optimal path (red)
        
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    
    # Mark Start and Goal clearly
    ax.text(start[1], start[0], 'S', color='green', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(goal[1], goal[0], 'G', color='darkblue', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Grid formatting
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# 5 Test cases required by the document
test_cases = [
    {
        "name": "Test Case 1: Simple Path",
        "grid": [
            "S . . . . . . .",
            ". . # # # . . .",
            ". . . . # . . .",
            ". # . . # . . .",
            ". # . . . . . .",
            ". . . . . . . G"
        ],
        "file": "testcase_1.png"
    },
    {
        "name": "Test Case 2: Multiple Possible Paths",
        "grid": [
            "S . . . .",
            ". # # # .",
            ". . . . .",
            ". # # # .",
            ". . . . G"
        ],
        "file": "testcase_2.png"
    },
    {
        "name": "Test Case 3: Narrow Passage",
        "grid": [
            "S . # . .",
            ". . # . .",
            "# . # . #",
            ". . . . .",
            ". . # . G"
        ],
        "file": "testcase_3.png"
    },
    {
        "name": "Test Case 4: Different Obstacle Arrangement",
        "grid": [
            "S # . . . .",
            ". # . # # .",
            ". . . # . .",
            "# # . # . #",
            ". . . . . G"
        ],
        "file": "testcase_4.png"
    },
    {
        "name": "Test Case 5: No Valid Path",
        "grid": [
            "S . . # .",
            ". . . # .",
            "# # # # .",
            ". . . # #",
            ". . . # G"
        ],
        "file": "testcase_5.png"
    }
]

# Clean up spaces in grid strings
for tc in test_cases:
    tc["grid"] = [row.replace(" ", "") for row in tc["grid"]]

# Execution loop
for tc in test_cases:
    print("---Search Result---")
    found, path, cost, nodes_visited, exec_time, explored, start, goal = a_star_search(tc["grid"])
    
    if found:
        print(f"{tc['name']} Path Found: YES")
        print("Path:")
        for coord in path:
            print(f"({coord[0]},{coord[1]})")
        print(f"Total Path Cost: {cost} Nodes Explored: {nodes_visited}")
        print(f"Execution Time: {exec_time:.6f} seconds Visualization saved: {tc['file']}")
    else:
        print(f"{tc['name']} Path Found: NO")
        print("No valid path exists between Start and Goal.")
        print(f"Nodes Explored: {nodes_visited}")
        print(f"Execution Time: {exec_time:.6f} seconds Visualization saved: {tc['file']}")
        
    visualize_and_save(tc["grid"], path, explored, start, goal, tc["file"], tc["name"])
    print()