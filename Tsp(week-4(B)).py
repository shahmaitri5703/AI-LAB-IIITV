import numpy as np
import matplotlib.pyplot as plt
import random
import math

def parse_tsp_file(filename):
   
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "NODE_COORD_SECTION":
                break
        for line in f:
            line = line.strip()
            if line == "EOF" or "SECTION" in line:
                break
            if line:
                parts = line.split()
                coords.append([float(parts[1]), float(parts[2])])
    print(f"Parsed {len(coords)} coordinates from {filename}.")
    return np.array(coords)

def calculate_total_distance(tour, distance_matrix):
  
    total_dist = 0
    num_cities = len(tour)
    for i in range(num_cities):
        total_dist += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
    return total_dist

def plot_tour(coords, tour, filename):
  
    plt.figure(figsize=(10, 8))
    tour_coords = coords[tour]
    tour_coords = np.vstack([tour_coords, tour_coords[0]])
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'r-')
    plt.plot(coords[:, 0], coords[:, 1], 'bo', markersize=3) 
    plt.title(f"TSP Solution for {filename}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

def simulated_annealing_tsp(coords, initial_temp, cooling_rate, stopping_temp):
    
    num_cities = len(coords)
    distance_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))
    current_solution = list(range(num_cities))
    random.shuffle(current_solution)
    current_cost = calculate_total_distance(current_solution, distance_matrix)
    best_solution = list(current_solution)
    best_cost = current_cost
    temperature = initial_temp

    while temperature > stopping_temp:
        i, j = sorted(random.sample(range(num_cities), 2))
        neighbor_solution = current_solution[:i] + current_solution[i:j+1][::-1] + current_solution[j+1:]
        neighbor_cost = calculate_total_distance(neighbor_solution, distance_matrix)
        cost_delta = neighbor_cost - current_cost
        
        if cost_delta < 0 or (temperature > 1e-9 and random.random() < math.exp(-cost_delta / temperature)):
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        
        if current_cost < best_cost:
            best_solution = list(current_solution)
            best_cost = current_cost
            
        temperature *= cooling_rate
        
    return best_solution, best_cost

if __name__ == '__main__':
  
    filenames = [
        "xqg237.tsp",
        "xql662.tsp",
        "xvb13584.tsp",
        "xrb14233.tsp",
        "xrh24104.tsp"
    ]
    results = {}

    INITIAL_TEMP = 1000
    COOLING_RATE = 0.999
    STOPPING_TEMP = 1e-3

    for tsp_file in filenames:
        print(f"\nSolving {tsp_file} ")
        try:
            coordinates = parse_tsp_file(tsp_file)
            
            print(f"Running Simulated Annealing for {len(coordinates)} cities...")
            
            final_solution, final_cost = simulated_annealing_tsp(
                coordinates, INITIAL_TEMP, COOLING_RATE, STOPPING_TEMP
            )
            
            results[tsp_file] = final_cost
            print(f"Found tour cost for {tsp_file}: {final_cost:.2f}")

       
            plot_tour(coordinates, final_solution, tsp_file)

        except FileNotFoundError:
            print(f"Error: Could not find {tsp_file}. Please ensure it's in the same folder.")
        except Exception as e:
            print(f"An error occurred while processing {tsp_file}: {e}")


    print("\n\nFinal Comparison of Results")
    for filename, cost in results.items():
        print(f"Problem: {filename.ljust(15)} | Your Found Cost: {cost:.2f}")