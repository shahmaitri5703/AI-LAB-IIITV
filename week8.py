# ===========================================================
# week8_gbike_modified.py
# Lab Assignment 8 – Problem 3: Modified Gbike Bicycle Rental
# Policy Iteration with free shuttle (loc1→loc2) + parking penalty
# IIIT Vadodara – CS367/659 Artificial Intelligence
# Authors: Maitri Shah, K. Siddharth Reddy, Ananya Saxena
# ===========================================================

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import pandas as pd
import time

# ----------------------- Configuration -----------------------
MAX_BIKES = 20
MAX_MOVE = 5
RENT_REWARD = 10
MOVE_COST = 2
FREE_FROM_1_TO_2 = 1        # first bike from loc1 → loc2 is free
PARKING_LIMIT = 10
PARKING_COST = 4
GAMMA = 0.9

# Poisson parameters
req_lambda = [3, 4]   # requests at loc1, loc2
ret_lambda = [3, 2]   # returns at loc1, loc2
POISSON_UPPER = 11    # truncate Poisson tail beyond this

# ----------------------- Poisson Cache -----------------------
poisson_cache = {}
def truncated_poisson_pmf(lam):
    if lam in poisson_cache:
        return poisson_cache[lam]
    probs = [poisson.pmf(i, lam) for i in range(POISSON_UPPER)]
    tail = 1.0 - sum(probs)
    probs.append(tail)
    poisson_cache[lam] = probs
    return probs

req_pmf_0 = truncated_poisson_pmf(req_lambda[0])
req_pmf_1 = truncated_poisson_pmf(req_lambda[1])
ret_pmf_0 = truncated_poisson_pmf(ret_lambda[0])
ret_pmf_1 = truncated_poisson_pmf(ret_lambda[1])

def actual_count(idx):
    return idx if idx < POISSON_UPPER else POISSON_UPPER

def iterate_demands_returns():
    for r0 in range(POISSON_UPPER + 1):
        p_r0 = req_pmf_0[r0]
        if p_r0 == 0: continue
        for r1 in range(POISSON_UPPER + 1):
            p_r1 = req_pmf_1[r1]
            if p_r1 == 0: continue
            for ret0 in range(POISSON_UPPER + 1):
                p_ret0 = ret_pmf_0[ret0]
                if p_ret0 == 0: continue
                for ret1 in range(POISSON_UPPER + 1):
                    p_ret1 = ret_pmf_1[ret1]
                    if p_ret1 == 0: continue
                    prob = p_r0 * p_r1 * p_ret0 * p_ret1
                    if prob > 0:
                        yield (r0, r1, ret0, ret1, prob)

# ----------------------- States & Actions -----------------------
states = [(i, j) for i in range(MAX_BIKES+1) for j in range(MAX_BIKES+1)]
state_to_index = {s: idx for idx, s in enumerate(states)}
index_to_state = {idx: s for s, idx in state_to_index.items()}
actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

# ----------------------- Precompute Transitions -----------------------
print("Precomputing transition probabilities and rewards... (this takes ~9-10 minutes)")
start_time = time.time()

P = {}  # (state_idx, action) → list of (prob, next_state_idx, expected_reward)

for s_idx, (n1, n2) in enumerate(states):
    for a in actions:
        moved = a  # positive = loc1 → loc2
        n1_after = n1 - moved
        n2_after = n2 + moved

        if not (0 <= n1_after <= MAX_BIKES and 0 <= n2_after <= MAX_BIKES):
            P[(s_idx, a)] = []
            continue

        # Move cost (free first bike loc1→loc2)
        move_cost = MOVE_COST * max(0, moved - FREE_FROM_1_TO_2) if moved > 0 else MOVE_COST * abs(moved)

        # Parking cost (after moving)
        parking_charge = PARKING_COST * ((n1_after > PARKING_LIMIT) + (n2_after > PARKING_LIMIT))
        immediate_cost = move_cost + parking_charge

        transitions = {}
        for req0_idx, req1_idx, ret0_idx, ret1_idx, prob in iterate_demands_returns():
            d0 = actual_count(req0_idx)
            d1 = actual_count(req1_idx)
            r0 = actual_count(ret0_idx)
            r1 = actual_count(ret1_idx)

            rentals0 = min(n1_after, d0)
            rentals1 = min(n2_after, d1)
            rental_reward = RENT_REWARD * (rentals0 + rentals1)

            end1 = min(n1_after - rentals0 + r0, MAX_BIKES)
            end2 = min(n2_after - rentals1 + r1, MAX_BIKES)
            next_state = (end1, end2)
            next_idx = state_to_index[next_state]

            total_reward = rental_reward - immediate_cost

            if next_idx not in transitions:
                transitions[next_idx] = [0.0, 0.0]
            transitions[next_idx][0] += prob
            transitions[next_idx][1] += prob * total_reward

        trans_list = []
        for nxt, (p_sum, rew_sum) in transitions.items():
            if p_sum > 0:
                exp_reward = rew_sum / p_sum
                trans_list.append((p_sum, nxt, exp_reward))
        P[(s_idx, a)] = trans_list

print(f"Precomputation completed in {time.time() - start_time:.1f} seconds")

# ----------------------- Policy Iteration -----------------------
num_states = len(states)
policy = np.zeros(num_states, dtype=int)

# Initialize with feasible actions
for s_idx in range(num_states):
    feasible = [a for a in actions if P[(s_idx, a)]]
    policy[s_idx] = feasible[0] if feasible else 0

V = np.zeros(num_states)
THETA = 1e-4

def policy_evaluation(policy, V):
    iteration = 0
    while True:
        delta = 0.0
        for s_idx in range(num_states):
            a = policy[s_idx]
            trans = P[(s_idx, a)]
            if not trans:
                v_new = 0
            else:
                v_new = sum(p * (r + GAMMA * V[next_s]) for p, next_s, r in trans)
            delta = max(delta, abs(V[s_idx] - v_new))
            V[s_idx] = v_new
        iteration += 1
        if delta < THETA:
            return iteration

def policy_improvement(policy, V):
    stable = True
    new_policy = policy.copy()
    for s_idx in range(num_states):
        old_a = policy[s_idx]
        best_a = old_a
        best_val = -np.inf

        for a in actions:
            trans = P[(s_idx, a)]
            if not trans: continue
            q = sum(p * (r + GAMMA * V[next_s]) for p, next_s, r in trans)
            if q > best_val:
                best_val = q
                best_a = a

        new_policy[s_idx] = best_a
        if best_a != old_a:
            stable = False
    return stable, new_policy

# Run Policy Iteration
print("Starting Policy Iteration...")
iteration = 0
while True:
    iteration += 1
    eval_iters = policy_evaluation(policy, V)
    stable, policy = policy_improvement(policy, V)
    print(f"Iteration {iteration} | Policy Eval iters: {eval_iters} | Stable: {stable}")
    if stable:
        break

print(f"\nPolicy Iteration converged in {iteration} iterations!")

# ----------------------- Reshape Results -----------------------
policy_grid = np.zeros((MAX_BIKES+1, MAX_BIKES+1), dtype=int)
value_grid = np.zeros((MAX_BIKES+1, MAX_BIKES+1))

for (n1, n2), idx in state_to_index.items():
    policy_grid[n1, n2] = policy[idx]
    value_grid[n1, n2] = V[idx]

# ----------------------- Display & Save -----------------------
print("\nOptimal Policy (rows = bikes at loc1, cols = bikes at loc2):")
print(policy_grid)

print("\nValue Function (rounded):")
print(np.round(value_grid, 1))

# Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
im1 = plt.imshow(policy_grid, origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
plt.title("Optimal Policy\n(Bikes moved: loc1 → loc2)")
plt.xlabel("Bikes at Location 2")
plt.ylabel("Bikes at Location 1")
plt.colorbar(im1, label="Action (negative = loc2 → loc1)")

plt.subplot(1, 2, 2)
im2 = plt.imshow(value_grid, origin='lower', cmap='viridis')
plt.title("Optimal Value Function")
plt.xlabel("Bikes at Location 2")
plt.ylabel("Bikes at Location 1")
plt.colorbar(im2, label="Value")

plt.tight_layout()
plt.show()

# Save to CSV
pd.DataFrame(policy_grid).to_csv("gbike_optimal_policy.csv", index=False)
pd.DataFrame(value_grid).to_csv("gbike_value_function.csv", index=False)
print("\nResults saved to:")
print("  → gbike_optimal_policy.csv")
print("  → gbike_value_function.csv")

print("\nDone! Submit this .py file + the two CSV files + plots.")