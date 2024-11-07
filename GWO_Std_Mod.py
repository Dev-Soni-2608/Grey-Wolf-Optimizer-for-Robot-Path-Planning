import numpy as np
import matplotlib.pyplot as plt

# Parameters
Pop_size = 10
Max_T = 75
Dim = 2   
Num_obstacles = 50
safe_distance = 5
bounds = np.array([[0, 100], [0, 100]])

def generate_start_and_goal(bounds, safe_distance):
    start = np.random.uniform(bounds[:, 0], bounds[:, 1])
    goal = np.random.uniform(bounds[:, 0], bounds[:, 1])
    return start, goal

# Generate obstacles with random placement
def generate_obstacles(num_obstacles, bounds, start, goal, safe_distance):
    obstacles = []
    while len(obstacles) < num_obstacles:
        obs = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, 2)).flatten()
        if (np.linalg.norm(obs - start) >= safe_distance and
            np.linalg.norm(obs - goal) >= safe_distance and
            all(np.linalg.norm(obs - other) >= 2*safe_distance for other in obstacles)):
            obstacles.append(obs)
    return np.array(obstacles)

# Fitness function for both GWO versions
def fitness_function(position, goal, obstacles, safe_distance):
    distance_to_goal = np.linalg.norm(position - goal)
    penalty = sum([1000 / dis_to_obs for obs in obstacles
                   if (dis_to_obs := np.linalg.norm(position - obs)) < safe_distance])
    return distance_to_goal + penalty

# Standard GWO implementation
def standard_gwo(Pop_size, Max_T, Dim, start, goal, bounds, obstacles, safe_distance):
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(Pop_size, Dim))
    alpha_pos, beta_pos, delta_pos = start.copy(), None, None
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    best_path, best_scores = [start.copy()], []

    for iter in range(Max_T):
        for i in range(Pop_size):
            fitness = fitness_function(positions[i, :], goal, obstacles, safe_distance)
            
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = alpha_score, alpha_pos
                alpha_score, alpha_pos = fitness, positions[i, :].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = fitness, positions[i, :].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i, :].copy()

        best_path.append(alpha_pos.copy())
        best_scores.append(alpha_score)

        for i in range(Pop_size):
            for j in range(Dim):
                r1_alpha, r2_alpha = np.random.rand(), np.random.rand()
                A1 = 2 * (2 - iter * (2 / Max_T)) * r1_alpha - (2 - iter * (2 / Max_T))
                C1 = 2 * r2_alpha
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                A2, C2 = 2 * (2 - iter * (2 / Max_T)) * np.random.rand() - (2 - iter * (2 / Max_T)), 2 * np.random.rand()
                D_beta = abs(C2 * beta_pos[j] - positions[i, j]) if beta_pos is not None else 0
                X2 = beta_pos[j] - A2 * D_beta

                A3, C3 = 2 * (2 - iter * (2 / Max_T)) * np.random.rand() - (2 - iter * (2 / Max_T)), 2 * np.random.rand()
                D_delta = abs(C3 * delta_pos[j] - positions[i, j]) if delta_pos is not None else 0
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

    return best_path, best_scores, alpha_pos

# Improved GWO implementation
def modified_gwo(Pop_size, Max_T, Dim, start, goal, bounds, obstacles, safe_distance):
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(Pop_size, Dim))
    alpha_pos, beta_pos, delta_pos = start.copy(), None, None
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    best_path, best_scores = [start.copy()], []

    for iter in range(Max_T):
        a = 2 - iter * (2 / Max_T)

        for i in range(Pop_size):
            fitness = fitness_function(positions[i, :], goal, obstacles, safe_distance)
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = alpha_score, alpha_pos
                alpha_score, alpha_pos = fitness, positions[i, :].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = fitness, positions[i, :].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i, :].copy()

        best_path.append(alpha_pos.copy())
        best_scores.append(alpha_score)

        for i in range(Pop_size):
            for j in range(Dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                A2, C2 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                D_beta = abs(C2 * beta_pos[j] - positions[i, j]) if beta_pos is not None else 0
                X2 = beta_pos[j] - A2 * D_beta

                A3, C3 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                D_delta = abs(C3 * delta_pos[j] - positions[i, j]) if delta_pos is not None else 0
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (0.6 * X1 + 0.3 * X2 + 0.1 * X3)

        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

    return best_path, best_scores, alpha_pos

start, goal = generate_start_and_goal(bounds, safe_distance)
obstacles = generate_obstacles(Num_obstacles, bounds, start, goal, safe_distance)

# Run both GWO versions
standard_path, standard_scores, standard_final_pos = standard_gwo(Pop_size, Max_T, Dim, start, goal, bounds, obstacles, safe_distance)
modified_path, modified_scores, modified_final_pos = modified_gwo(Pop_size, Max_T, Dim, start, goal, bounds, obstacles, safe_distance)

# Plot final paths
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Standard GWO - Final Path")
plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
plt.plot(obstacles[:, 0], obstacles[:, 1], 'ks', markersize=4, label='Obstacles')
standard_path_np = np.array(standard_path)
plt.plot(standard_path_np[:, 0], standard_path_np[:, 1], 'c-', linewidth=1.5, label='Path')
plt.scatter(standard_path_np[:, 0], standard_path_np[:, 1], color='c', s=20, marker='o')
plt.plot(standard_final_pos[0], standard_final_pos[1], 'c*', markersize=12, label='Final Alpha Position')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("modified GWO - Final Path")
plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
plt.plot(obstacles[:, 0], obstacles[:, 1], 'ks', markersize=4, label='Obstacles')
modified_path_np = np.array(modified_path)
plt.plot(modified_path_np[:, 0], modified_path_np[:, 1], 'm-', linewidth=1.5, label='Path')
plt.scatter(modified_path_np[:, 0], modified_path_np[:, 1], color='m', s=20, marker='o')
plt.plot(modified_final_pos[0], modified_final_pos[1], 'm*', markersize=12, label='Final Alpha Position')
plt.legend()
plt.show()

# Combined convergence curves
plt.figure()
plt.plot(range(Max_T), standard_scores, 'b-', label='Standard GWO')
plt.plot(range(Max_T), modified_scores, 'r-', label='modified GWO')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Score')
plt.title('Convergence Curve')
plt.legend()
plt.show()

# Calculate best path length
standard_path_Len = sum(np.linalg.norm(standard_path_np[i] - standard_path_np[i + 1])
                       for i in range(len(standard_path_np) - 1))

print(f'Standard Path Length: {standard_path_Len}')

modified_path_Len = sum(np.linalg.norm(modified_path_np[i] - modified_path_np[i + 1])
                       for i in range(len(modified_path_np) - 1))

print(f'modified Path Length: {modified_path_Len}')