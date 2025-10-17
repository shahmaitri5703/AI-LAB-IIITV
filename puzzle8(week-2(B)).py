from collections import deque



def get_valid_moves(state):
   
    moves = []
    empty_tile = state.index(0)
    row, col = divmod(empty_tile, 3)
    if row > 0: moves.append(empty_tile - 3)  
    if row < 2: moves.append(empty_tile + 3)  
    if col > 0: moves.append(empty_tile - 1)  
    if col < 2: moves.append(empty_tile + 1)  
    return moves

def make_move(state, new_pos):
    
    new_state = list(state)
    empty_tile = new_state.index(0)
    new_state[empty_tile], new_state[new_pos] = new_state[new_pos], new_state[empty_tile]
    return tuple(new_state)

def is_goal_state(state, goal_state):
    
    return state == goal_state


class Node:
    def __init__(self, state, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.depth = depth

def iterative_deepening_search(start_state, goal_state, max_depth=50):
   
    def dls(node, goal_state, limit):
        if is_goal_state(node.state, goal_state):
            return node
        if limit == 0:
            return None
        for new_pos in get_valid_moves(node.state):
            child_state = make_move(node.state, new_pos)
            child = Node(child_state, parent=node, depth=node.depth + 1)
            result = dls(child, goal_state, limit - 1)
            if result is not None:
                return result
        return None

    for depth_limit in range(max_depth):
        result = dls(Node(start_state), goal_state, depth_limit)
        if result is not None:
            return result
    return None

def reconstruct_path(node):
    
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]


if __name__ == "__main__":
    try:
        start = (1, 2, 3, 4, 0, 5, 6, 7, 8)  
        goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)   

        print("Running 8-Puzzle Iterative Deepening Search...")
        result_node = iterative_deepening_search(start, goal, max_depth=20)
        
        if result_node:
            solution_path = reconstruct_path(result_node)
            print(f"\nSolution found in {len(solution_path)-1} moves:")
            for i, state in enumerate(solution_path, 1):
                print(f"\nMove {i}:")
                print("---------")
                print(f"{state[0]} {state[1]} {state[2]}")
                print(f"{state[3]} {state[4]} {state[5]}")
                print(f"{state[6]} {state[7]} {state[8]}")
                print("---------")
        else:
            print("No solution found within depth limit.")
    except Exception as e:
        print(f"An error occurred: {e}")
