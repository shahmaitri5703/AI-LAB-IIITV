import heapq
import time


INITIAL_BOARD_STATE = [
    [-1, -1, 1, 1, 1, -1, -1],
    [-1, -1, 1, 1, 1, -1, -1],
    [ 1,  1, 1, 1, 1,  1,  1],
    [ 1,  1, 1, 0, 1,  1,  1],
    [ 1,  1, 1, 1, 1,  1,  1],
    [-1, -1, 1, 1, 1, -1, -1],
    [-1, -1, 1, 1, 1, -1, -1],
]


FINAL_SIMPLE_BOARD = [
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],  
    [-1, -1, -1, 1, -1, -1, -1],  
    [-1, -1, -1, 0, -1, -1, -1],  
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
]

BOARD_CENTER = (3, 3)

def get_marbles(board_state):

    marbles = set()
    for r, row in enumerate(board_state):
        for c, val in enumerate(row):
            if val == 1:
                marbles.add((r, c))
    return frozenset(marbles)

def is_valid_pos(r, c):

    return 0 <= r < 7 and 0 <= c < 7 and INITIAL_BOARD_STATE[r][c] != -1

def generate_moves(marbles):

    moves = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for r, c in marbles:
        for dr, dc in directions:
            jumped_r, jumped_c = r + dr, c + dc
            dest_r, dest_c = r + 2 * dr, c + 2 * dc
            
            if (jumped_r, jumped_c) in marbles and \
               is_valid_pos(dest_r, dest_c) and \
               (dest_r, dest_c) not in marbles:
                
                new_marbles = set(marbles)
                new_marbles.remove((r, c))
                new_marbles.remove((jumped_r, jumped_c))
                new_marbles.add((dest_r, dest_c))
                moves.append(frozenset(new_marbles))
    return moves



def h1_marble_count(marbles):
 
    return len(marbles) - 1

def h2_avg_distance_from_center(marbles):

    if not marbles:
        return 0
    total_dist = sum(abs(r - BOARD_CENTER[0]) + abs(c - BOARD_CENTER[1]) for r, c in marbles)
    return total_dist / len(marbles)
    


def solve_solitaire(initial_marbles, heuristic_func, algorithm):
    

    NODE_LIMIT = 500000 
    
    initial_g_cost = 0
    initial_state = initial_marbles
    
    priority = 0
    if algorithm == 'GBFS':
        priority = heuristic_func(initial_state)
    elif algorithm == 'A*':
        priority = initial_g_cost + heuristic_func(initial_state)
    elif algorithm == 'UCS':
        priority = initial_g_cost

    frontier = [(priority, initial_g_cost, initial_state, [])]
    explored = set()
    nodes_expanded = 0

    while frontier:

        if nodes_expanded > NODE_LIMIT:
            print(f"⚠️  Node limit of {NODE_LIMIT} exceeded. Stopping search.")
            break

        _, g_cost, current_marbles, path = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_marbles in explored:
            continue
        
        explored.add(current_marbles)
        
        if len(current_marbles) == 1 and BOARD_CENTER in current_marbles:

            print(f" Solution Found for {algorithm}!")





            print(f"  - Path Length: {g_cost} moves")
            print(f"  - Nodes Expanded: {nodes_expanded}")
            return path + [current_marbles]

        for next_state in generate_moves(current_marbles):
            if next_state not in explored:
                new_g_cost = g_cost + 1
                new_priority = 0
                if algorithm == 'GBFS':
                    new_priority = heuristic_func(next_state)
                elif algorithm == 'A*':
                    new_priority = new_g_cost + heuristic_func(next_state)
                elif algorithm == 'UCS':
                    new_priority = new_g_cost
                heapq.heappush(frontier, (new_priority, new_g_cost, next_state, path + [current_marbles]))
                
    print(f" No solution found for {algorithm} (or limit was reached).")
    return None



if __name__ == '__main__':
    print("="*50)
    print(" MARBLE SOLITAIRE ")
    print("="*50)

  
    start_marbles_full = get_marbles(INITIAL_BOARD_STATE)
    print("\n Running Greedy Best-First Search (GBFS) with h2 on FULL board...")
    start_time = time.time()
    solve_solitaire(start_marbles_full, h2_avg_distance_from_center, 'GBFS')
    print(f"Time taken: {time.time() - start_time:.4f} seconds")

    start_marbles_simple = get_marbles(FINAL_SIMPLE_BOARD)
    print("\n Running A* Search with h1 on a FINAL SIMPLE board...")
    start_time = time.time()
    solve_solitaire(start_marbles_simple, h1_marble_count, 'A*')
    print(f"Time taken: {time.time() - start_time:.4f} seconds")