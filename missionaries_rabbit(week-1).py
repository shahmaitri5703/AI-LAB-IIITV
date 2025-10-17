from collections import deque



def is_valid_mc_state(state):
    m, c, b = state
  
    if m < 0 or c < 0 or m > 3 or c > 3:
        return False
    if (m > 0 and m < c) or (3 - m > 0 and 3 - m < 3 - c):
        return False
    return True

def get_mc_successors(state):
    m, c, b = state
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]  
    successors = []
    for move_m, move_c in moves:
        if b == 1:  
            new_state = (m - move_m, c - move_c, 0)
            direction = "left-to-right"
        else: 
            new_state = (m + move_m, c + move_c, 1)
            direction = "right-to-left"
        if is_valid_mc_state(new_state):
            successors.append((new_state, (move_m, move_c, direction)))
    return successors

def bfs_mc():
    start = (3, 3, 1)  
    goal = (0, 0, 0)
    queue = deque([(start, [])])
    visited = set([start])
    
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for succ, move in get_mc_successors(state):
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path + [move]))
    return None

def dfs_mc():
    start = (3, 3, 1)
    goal = (0, 0, 0)
    stack = [(start, [])]
    visited = set()
    
    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        if state in visited:
            continue
        visited.add(state)
        for succ, move in get_mc_successors(state):
            if succ not in visited:
                stack.append((succ, path + [move]))
    return None


def format_rabbit_state(state):
    return "".join(state)

def is_valid_rabbit_state(state):
   
    return True

def get_rabbit_successors(state):
    successors = []
    length = len(state)
    empty_idx = state.index('_')
    
    for i in range(length):
        if state[i] == '_':
            continue
        if state[i] == 'E':  
            if i + 1 == empty_idx:
                new_state = state[:]
                new_state[i], new_state[empty_idx] = new_state[empty_idx], new_state[i]
                successors.append(new_state)
            elif i + 2 == empty_idx and i + 1 < length and state[i + 1] != '_':
                new_state = state[:]
                new_state[i], new_state[empty_idx] = new_state[empty_idx], new_state[i]
                successors.append(new_state)
        elif state[i] == 'W':  
            if i - 1 == empty_idx:
                new_state = state[:]
                new_state[i], new_state[empty_idx] = new_state[empty_idx], new_state[i]
                successors.append(new_state)
            elif i - 2 == empty_idx and i - 1 >= 0 and state[i - 1] != '_':
                new_state = state[:]
                new_state[i], new_state[empty_idx] = new_state[empty_idx], new_state[i]
                successors.append(new_state)
    return successors

def bfs_rabbit():
    start = ['E', 'E', 'E', '_', 'W', 'W', 'W']
    goal = ['W', 'W', 'W', '_', 'E', 'E', 'E']
    start_t = tuple(start)
    goal_t = tuple(goal)
    queue = deque([(start_t, [start_t])])
    visited = set([start_t])
    
    while queue:
        state, path = queue.popleft()
        if state == goal_t:
            return path
        for succ in get_rabbit_successors(list(state)):
            succ_t = tuple(succ)
            if succ_t not in visited:
                visited.add(succ_t)
                queue.append((succ_t, path + [succ_t]))
    return None

def dfs_rabbit():
    start = ['E', 'E', 'E', '_', 'W', 'W', 'W']
    goal = ['W', 'W', 'W', '_', 'E', 'E', 'E']
    start_t = tuple(start)
    goal_t = tuple(goal)
    stack = [(start_t, [start_t])]
    visited = set()
    
    while stack:
        state, path = stack.pop()
        if state == goal_t:
            return path
        if state in visited:
            continue
        visited.add(state)
        for succ in get_rabbit_successors(list(state)):
            succ_t = tuple(succ)
            if succ_t not in visited:
                stack.append((succ_t, path + [succ_t]))
    return None

def print_path(path, problem_name, is_mc=False):
    if not path:
        print(f"\n{problem_name}: No solution found")
        return
    print(f"\n{problem_name} Solution ({len(path)-1} moves):")
    if is_mc:
        for i, move in enumerate(path):
            move_m, move_c, direction = move
            print(f"Step {i+1}: Move {move_m} missionaries, {move_c} cannibals {direction}")
    else:
        for i, state in enumerate(path):
            print(f"Step {i+1}: {format_rabbit_state(state)}")

if __name__ == "__main__":
    try:
       
        print("Running Missionaries and Cannibals...")
        bfs_mc_path = bfs_mc()
        print_path(bfs_mc_path, "Missionaries and Cannibals BFS", is_mc=True)
        
        dfs_mc_path = dfs_mc()
        print_path(dfs_mc_path, "Missionaries and Cannibals DFS", is_mc=True)
        
       
        print("\nRunning Rabbit Leap...")
        bfs_rabbit_path = bfs_rabbit()
        print_path(bfs_rabbit_path, "Rabbit Leap BFS")
        
        dfs_rabbit_path = dfs_rabbit()
        print_path(dfs_rabbit_path, "Rabbit Leap DFS")
    except Exception as e:
        print(f"An error occurred: {e}")
