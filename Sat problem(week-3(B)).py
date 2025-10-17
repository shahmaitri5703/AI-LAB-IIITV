import random
import itertools
import time
import matplotlib.pyplot as plt

def generate_k_sat(n, m, k):
    
    if k > n:
        raise ValueError("k cannot be greater than n for distinct variables.")
    
    formula = []
    variables = list(range(1, n + 1))
    
    for _ in range(m):
        clause_vars = random.sample(variables, k)
        clause = [var if random.choice([True, False]) else -var for var in clause_vars]
        formula.append(clause)
        
    return formula

def h1_satisfied_clauses(formula, assignment):
  
    satisfied_count = 0
    for clause in formula:
        for literal in clause:
            var_index = abs(literal) - 1
            if (literal > 0 and assignment[var_index]) or (literal < 0 and not assignment[var_index]):
                satisfied_count += 1
                break
    return satisfied_count

def hill_climbing(formula, n, max_restarts=50, max_steps=1000):
    
    for restart in range(1, max_restarts + 1):
        current_assignment = [random.choice([True, False]) for _ in range(n)]
        score_history = [h1_satisfied_clauses(formula, current_assignment)]
        
        for step in range(1, max_steps):
            current_score = score_history[-1]
            if current_score == len(formula):
                print(f"Hill-Climbing: Solution found on restart {restart}!")
                print(f"  - Steps (L): {step}")
                return current_assignment, score_history

            best_neighbor, best_neighbor_score = None, current_score
            for i in range(n):
                neighbor = list(current_assignment)
                neighbor[i] = not neighbor[i]
                neighbor_score = h1_satisfied_clauses(formula, neighbor)
                
               
                if neighbor_score > best_neighbor_score:
                    best_neighbor, best_neighbor_score = neighbor, neighbor_score
            
            if best_neighbor is None:
                break 
            
            current_assignment = best_neighbor
            score_history.append(best_neighbor_score)
            
    print("Hill-Climbing: No solution found after all restarts.")
    return None, []

def beam_search_with_restarts(formula, n, beam_width, max_restarts=20, max_iterations=500):
   
    for restart in range(1, max_restarts + 1):
        beam = [[random.choice([True, False]) for _ in range(n)] for _ in range(beam_width)]
        
        for iteration in range(max_iterations):
            for assignment in beam:
                if h1_satisfied_clauses(formula, assignment) == len(formula):
                    print(f"Beam Search (b={beam_width}): Solution found on restart {restart} in iteration {iteration + 1}!")
                    return

            all_neighbors = []
            for assignment in beam:
                for i in range(n):
                    neighbor = list(assignment)
                    neighbor[i] = not neighbor[i]
                    all_neighbors.append(neighbor)
            
            unique_neighbors = [list(x) for x in set(tuple(x) for x in all_neighbors)]
            neighbor_scores = []
            for neighbor in unique_neighbors:
                score = h1_satisfied_clauses(formula, neighbor)
                neighbor_scores.append((score, neighbor))
            
            neighbor_scores.sort(key=lambda x: x[0], reverse=True)
            
            if not neighbor_scores or neighbor_scores[0][0] <= h1_satisfied_clauses(formula, beam[0]):
                break 
            
            beam = [item[1] for item in neighbor_scores[:beam_width]]

    print(f"Beam Search (b={beam_width}): No solution found after all restarts.")

def variable_neighborhood_descent_with_restarts(formula, n, max_restarts=10, max_steps=500):
  
    num_clauses = len(formula)
    
    for restart in range(1, max_restarts + 1):
        s = [random.choice([True, False]) for _ in range(n)]
        for step in range(1, max_steps):
            current_score = h1_satisfied_clauses(formula, s)
            if current_score == num_clauses:
                print(f"VND: Solution found on restart {restart}!")
                print(f"  - Steps (L): {step}")
                return
            
            k = 1
            moved = False
            while k <= 2 and not moved:
                best_neighbor_in_Nk, best_neighbor_score = None, current_score
                indices_to_flip_combinations = itertools.combinations(range(n), k)
                
                for combo in indices_to_flip_combinations:
                    neighbor = list(s)
                    for index in combo:
                        neighbor[index] = not neighbor[index]
                    score = h1_satisfied_clauses(formula, neighbor)
                    if score > best_neighbor_score:
                        best_neighbor_in_Nk, best_neighbor_score = neighbor, score
                
                if best_neighbor_in_Nk is not None:
                    s = best_neighbor_in_Nk
                    moved = True
                else:
                    k += 1
    
    print("VND: No solution found after all restarts.")

if __name__ == '__main__':
    print("="*50)
    print("3-SAT SOLVING ")
    print("="*50)

  
    n_vars = 20
    m_clauses = int(4.2 * n_vars) 
    k_literals = 3
    
    problem_3sat = generate_k_sat(n_vars, m_clauses, k_literals)
    print(f"\nGenerated a 3-SAT Problem (n={n_vars}, m={m_clauses}, k={k_literals})")
    print("-" * 50)
    
    print("\nSolving with Hill-Climbing")
    start_time = time.time()
    solution, history = hill_climbing(problem_3sat, n_vars)
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    
  
    if solution:
        print("Generating plot for Hill-Climbing...")
        plt.figure(figsize=(10, 6))
        plt.plot(history, marker='o', linestyle='-', markersize=4)
        plt.title("Hill Climbing Convergence for 3-SAT")
        plt.xlabel("Iteration Number")
        plt.ylabel("Number of Satisfied Clauses")
        plt.grid(True)
        plt.show()

    print("\nSolving with Beam Search with Restarts (width=5) ")
    start_time = time.time()
    beam_search_with_restarts(problem_3sat, n_vars, beam_width=5)
    print(f"Time taken: {time.time() - start_time:.4f} seconds")

    print("\nSolving with Variable Neighborhood Descent with Restarts ")
    start_time = time.time()
    variable_neighborhood_descent_with_restarts(problem_3sat, n_vars)
    print(f"Time taken: {time.time() - start_time:.4f} seconds")