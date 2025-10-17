import numpy as np
import matplotlib.pyplot as plt
import random
import math

rajasthan_coords = np.array([
    [26.91, 75.78], [24.58, 73.68], [25.31, 74.64], [27.17, 78.00],
    [28.61, 77.20], [26.23, 73.02], [26.44, 74.63], [25.13, 75.84],
    [24.89, 74.62], [27.65, 76.63], [25.91, 73.25], [26.56, 75.81],
    [24.91, 75.83], [25.53, 73.21], [25.75, 75.80], [27.19, 75.92],
    [26.14, 75.82], [25.34, 74.64], [25.04, 74.52], [24.58, 73.68]
])

def calculate_total_distance(tour, distance_matrix):
    total_dist = 0
    num_cities = len(tour)
    for i in range(num_cities):
        total_dist += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
    return total_dist

def plot_tour(coords, tour, title):
    plt.figure(figsize=(10, 8))
    tour_coords = coords[tour]
    tour_coords = np.vstack([tour_coords, tour_coords[0]])
    plt.plot(tour_coords[:, 1], tour_coords[:, 0], 'r-')
    plt.plot(coords[:, 1], coords[:, 0], 'bo')
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
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
            best_cost = best_cost
            
        temperature *= cooling_rate
        
    return best_solution, best_cost

if __name__ == '__main__':
    print("the Rajasthan TSP ")
    
    INITIAL_TEMP = 1000
    COOLING_RATE = 0.9995
    STOPPING_TEMP = 1e-4

    print(f"Running Simulated Annealing for {len(rajasthan_coords)} cities...")
    
    final_solution, final_cost = simulated_annealing_tsp(
        rajasthan_coords, INITIAL_TEMP, COOLING_RATE, STOPPING_TEMP
    )
    
    print(f"Found tour with cost: {final_cost:.2f}")
    print(f" Best route (city order): {' -> '.join(map(str, final_solution))}")
    
    plot_tour(rajasthan_coords, final_solution, "TSP Solution for Rajasthan")