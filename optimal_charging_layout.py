import numpy as np
import math
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# --------------------------
# Parameter Initialization
# --------------------------
ALPHA = 0.1  # Temporal weight coefficient
BETA = 0.01  # Spatial weight coefficient
D_MAX = 1.5  # Maximum distance in km for user convenience constraint
GRID_VARIANCE_LIMIT_RATIO = 0.3  # Ratio for grid safety constraint


# --------------------------
# Similarity Calculation
# --------------------------
def calculate_similarity(t_i, t_j, x_i, x_j):
    """
    Calculate spatiotemporal similarity using the given formula:
    Similarity = 1 / (1 + α|t_i - t_j| + β||x_i - x_j||₂)
    """
    temporal_diff = abs(t_i - t_j)
    spatial_diff = distance.euclidean(x_i, x_j)
    similarity = 1 / (1 + ALPHA * temporal_diff + BETA * spatial_diff)
    return similarity


# --------------------------
# Wavelet-STDBSCAN Algorithm
# --------------------------
class WaveletSTDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def wavelet_transform(self, data, level=1):
        """Simplified wavelet transform for denoising and feature extraction"""
        transformed = np.zeros_like(data)
        for i in range(1, len(data) - 1):
            transformed[i] = (data[i - 1] + 2 * data[i] + data[i + 1]) / 4
        return transformed

    def fit_predict(self, time_series, locations):
        """
        Fit the model and predict clusters
        time_series: temporal data for each point
        locations: spatial coordinates for each point
        """
        # Apply wavelet transform to time series data
        denoised_ts = self.wavelet_transform(time_series)

        # Calculate similarity matrix
        n_points = len(denoised_ts)
        similarity_matrix = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                similarity_matrix[i, j] = calculate_similarity(
                    denoised_ts[i], denoised_ts[j],
                    locations[i], locations[j]
                )

        # Convert similarity to distance for DBSCAN
        distance_matrix = 1 - similarity_matrix

        # Perform clustering
        clusters = self.dbscan.fit_predict(distance_matrix)
        return clusters


# --------------------------
# Constraint Functions
# --------------------------
def check_user_convenience_constraint(charging_piles, user_locations, user_clusters):
    """
    Check if the user convenience constraint is satisfied:
    (1/N) * ΣΣ min||x_i - x_j|| ≤ D_max
    Handle DBSCAN noise points (users with label -1)
    """
    total_distance = 0.0
    valid_user_count = 0  # Count of non-noise users
    n_users = len(user_locations)

    for i in range(n_users):
        cluster_id = user_clusters[i]
        if cluster_id == -1:
            continue  # Skip noise users
        user_loc = user_locations[i]
        # Calculate minimum distance to any charging pile
        min_dist = min(distance.euclidean(user_loc, pile_loc) for pile_loc in charging_piles)
        total_distance += min_dist
        valid_user_count += 1

    if valid_user_count == 0:
        avg_distance = float('inf')
    else:
        avg_distance = total_distance / valid_user_count
    return avg_distance <= D_MAX, avg_distance


def check_grid_safety_constraint(ev_load_profile, grid_capacity):
    """
    Check if the grid safety constraint is satisfied using second-moment constraint:
    (1/T) * Σ(P_EV(t) - P̄_EV)² ≤ (0.3P̄_grid)²
    """
    t = len(ev_load_profile)
    avg_ev_load = np.mean(ev_load_profile)

    # Calculate variance of EV charging load
    load_variance = np.sum([(p - avg_ev_load) ** 2 for p in ev_load_profile]) / t

    # Calculate maximum allowed variance
    max_allowed_variance = (GRID_VARIANCE_LIMIT_RATIO * grid_capacity) ** 2

    return load_variance <= max_allowed_variance, load_variance


# --------------------------
# NSGA-III Implementation
# --------------------------
def generate_reference_points(n_obj, n_pop, priorities):
    """
    Generate proper 2D reference points array for NSGA-III
    """
    ref_points = []
    if n_obj == 3:  # For 3-objective optimization
        # Create a grid of reference points
        for i in range(n_obj + 1):
            for j in range(n_obj + 1 - i):
                k = n_obj - i - j
                if k >= 0:
                    ref_points.append([i, j, k])

        # Normalize reference points
        ref_points = np.array(ref_points, dtype=np.float64)
        ref_points = ref_points / np.sum(ref_points, axis=1)[:, np.newaxis]

        # Ensure no more reference points than population size
        if len(ref_points) > n_pop:
            ref_points = ref_points[:n_pop]
        return ref_points
    else:
        # Simple reference points generation for other number of objectives
        sum_priorities = sum(priorities)
        base_point = [p / sum_priorities for p in priorities]
        ref_points = [base_point.copy() for _ in range(min(n_pop, 100))]

        # Add small variation to reference points
        for i in range(len(ref_points)):
            for j in range(n_obj):
                ref_points[i][j] += random.uniform(-0.1, 0.1)
            # Re-normalize
            ref_points[i] = [p / sum(ref_points[i]) for p in ref_points[i]]
        return np.array(ref_points)


def initialize_charging_piles(n_piles, area_bounds):
    """Initialize charging pile locations randomly within given area bounds"""
    piles = []
    for _ in range(n_piles):
        x = random.uniform(area_bounds[0][0], area_bounds[0][1])
        y = random.uniform(area_bounds[1][0], area_bounds[1][1])
        piles.append((x, y))
    return piles


def objective_function(charging_piles, user_locations, user_clusters,
                       ev_load_profile, grid_capacity):
    """
    Multi-objective function to optimize:
    1. Minimize average distance (improve convenience)
    2. Minimize load variance (improve grid safety)
    3. Minimize number of charging piles (reduce cost)
    """
    # Objective 1: Minimize average distance
    _, avg_distance = check_user_convenience_constraint(
        charging_piles, user_locations, user_clusters)

    # Objective 2: Minimize load variance
    _, load_variance = check_grid_safety_constraint(
        ev_load_profile, grid_capacity)

    # Objective 3: Minimize number of charging piles
    n_piles = len(charging_piles)

    return (avg_distance, load_variance, n_piles)


def nsga3_optimization(user_locations, user_clusters, ev_load_profile,
                       grid_capacity, area_bounds, n_generations=50,
                       pop_size=100, n_obj=3, priorities=[1, 1, 1]):
    """NSGA-III optimization for charging pile layout"""
    # Reset DEAP's Fitness and Individual to avoid redefinition errors
    if 'FitnessMin' in dir(creator):
        del creator.FitnessMin
    if 'Individual' in dir(creator):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator: number of charging piles (5-20)
    toolbox.register("attr_n_piles", random.randint, 5, 20)

    # Structure initializers
    def init_individual():
        n_piles = toolbox.attr_n_piles()
        return creator.Individual(initialize_charging_piles(n_piles, area_bounds))

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    def evaluate(individual):
        return objective_function(
            individual, user_locations, user_clusters,
            ev_load_profile, grid_capacity
        )

    toolbox.register("evaluate", evaluate)

    # Custom crossover operator (swap part of charging piles between individuals)
    def mate(ind1, ind2):
        if random.random() < 0.5 and len(ind1) > 0 and len(ind2) > 0:
            split_idx = min(len(ind1), len(ind2)) // 2
            ind1[split_idx:], ind2[split_idx:] = ind2[split_idx:], ind1[split_idx:]
        return ind1, ind2

    # Custom mutation operator (modify location/add/remove charging piles)
    def mutate(individual):
        if len(individual) > 0:
            # 20% chance to mutate a charging pile's location
            if random.random() < 0.2:
                idx = random.randint(0, len(individual) - 1)
                x = random.uniform(area_bounds[0][0], area_bounds[0][1])
                y = random.uniform(area_bounds[1][0], area_bounds[1][1])
                individual[idx] = (x, y)
            # 10% chance to add a charging pile (up to 20)
            if random.random() < 0.1 and len(individual) < 20:
                x = random.uniform(area_bounds[0][0], area_bounds[0][1])
                y = random.uniform(area_bounds[1][0], area_bounds[1][1])
                individual.append((x, y))
            # 10% chance to remove a charging pile (at least 5)
            if random.random() < 0.1 and len(individual) > 5:
                del individual[random.randint(0, len(individual) - 1)]
        return individual,

    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)

    # Generate reference points for NSGA-III
    ref_points = generate_reference_points(n_obj, pop_size, priorities)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    # Run optimization
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # Hall of Fame to store the best individual

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(
        pop, toolbox, mu=pop_size, lambda_=pop_size,
        cxpb=0.7, mutpb=0.3, ngen=n_generations,
        stats=stats, halloffame=hof, verbose=True
    )

    return hof[0], pop


# --------------------------
# Main Execution
# --------------------------
def main():
    # Generate sample data
    np.random.seed(42)
    n_users = 100
    user_locations = np.random.rand(n_users, 2) * 10  # Users within 10x10 km area
    time_series = np.random.rand(n_users) * 24  # Random time data (0-24 hours)

    # Cluster users using Wavelet-STDBSCAN (tune eps and min_samples for better clustering)
    st_dbscan = WaveletSTDBSCAN(eps=0.5, min_samples=3)
    user_clusters = st_dbscan.fit_predict(time_series, user_locations)

    # Generate sample load profile (24 hours)
    ev_load_profile = np.random.normal(50, 15, 24)  # kW load
    grid_capacity = 200  # kW

    # Define area bounds for charging piles (same as user area)
    area_bounds = [(0, 10), (0, 10)]

    # Run optimization
    optimal_piles, population = nsga3_optimization(
        user_locations, user_clusters,
        ev_load_profile, grid_capacity,
        area_bounds, n_generations=30, pop_size=50
    )

    # Print optimal charging pile locations
    print("\nOptimal Charging Pile Locations:")
    for i, loc in enumerate(optimal_piles):
        print(f"Pile {i + 1}: ({loc[0]:.2f}, {loc[1]:.2f}) km")

    # Check constraint satisfaction
    conv_satisfied, avg_dist = check_user_convenience_constraint(
        optimal_piles, user_locations, user_clusters)
    grid_satisfied, load_var = check_grid_safety_constraint(
        ev_load_profile, grid_capacity)

    print(f"\nUser convenience constraint satisfied: {conv_satisfied}")
    print(f"Average distance: {avg_dist:.2f} km (Max allowed: {D_MAX} km)")
    print(f"Grid safety constraint satisfied: {grid_satisfied}")
    print(f"Load variance: {load_var:.2f} (Max allowed: {(GRID_VARIANCE_LIMIT_RATIO * grid_capacity) ** 2:.2f})")

    # Visualize results (differentiate clustered users and noise users)
    noise_users = user_locations[user_clusters == -1]  # DBSCAN noise users
    clustered_users = user_locations[user_clusters != -1]  # Users in clusters
    clustered_labels = user_clusters[user_clusters != -1]  # Cluster labels

    plt.figure(figsize=(10, 8))
    # Plot clustered users
    scatter_clustered = plt.scatter(
        clustered_users[:, 0], clustered_users[:, 1],
        c=clustered_labels, cmap='viridis', alpha=0.6,
        label='Clustered Users'
    )
    # Plot noise users if exist
    if len(noise_users) > 0:
        plt.scatter(
            noise_users[:, 0], noise_users[:, 1],
            c='gray', marker='.', alpha=0.6,
            label='Noise Users'
        )
    # Plot charging piles
    plt.scatter(
        [p[0] for p in optimal_piles], [p[1] for p in optimal_piles],
        marker='X', s=200, c='red',
        label='Charging Piles'
    )
    plt.title('Optimal Charging Pile Layout')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.legend()
    plt.grid(True)
    plt.colorbar(scatter_clustered, label='Cluster ID')  # Add colorbar for cluster IDs

    # Save figure as PDF to the specified path
    pdf_save_path = r"E:\0ECJTU\0ESSAY\arxiv\MSO\code_MSO\optimal_charging_layout.pdf"
    plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight')
    print(f"\nFigure saved as PDF to: {pdf_save_path}")

    plt.show()


if __name__ == "__main__":
    main()