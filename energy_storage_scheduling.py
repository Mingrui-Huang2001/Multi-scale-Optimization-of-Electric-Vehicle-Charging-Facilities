import numpy as np
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Try import of pulp, if unavailable prompt user
try:
    import pulp
except ImportError:
    print("请先安装PuLP库: pip install pulp")
    exit()

# Parameters
T = 24  # Time slots
N = 10  # Number of stations
np.random.seed(42)
demand = np.random.uniform(50, 100, (T, N))  # kWh demand
price = np.array([0.3 if h < 8 or h >= 20 else 0.8 for h in range(T)])
capacity = 50  # kWh
P_max = 25  # kW max charge/discharge

# 1. MILP Model
model = pulp.LpProblem("Storage_Scheduling", pulp.LpMinimize)
grid = pulp.LpVariable.dicts("grid", (range(T), range(N)), lowBound=0)
charge = pulp.LpVariable.dicts("charge", (range(T), range(N)), lowBound=0, upBound=P_max)
discharge = pulp.LpVariable.dicts("discharge", (range(T), range(N)), lowBound=0, upBound=P_max)
soc = pulp.LpVariable.dicts("soc", (range(T+1), range(N)), lowBound=0, upBound=capacity)
peak = pulp.LpVariable("peak", lowBound=0)
alpha = 0.5

# Objective
model += pulp.lpSum(grid[t][i]*price[t] for t in range(T) for i in range(N)) + alpha*peak

# Constraints
for i in range(N):
    model += soc[0][i] == capacity/2
    for t in range(T):
        model += grid[t][i] + discharge[t][i] >= demand[t][i]
        model += soc[t+1][i] == soc[t][i] + charge[t][i] - discharge[t][i]
        model += grid[t][i] <= peak

# Solve
model.solve(pulp.PULP_CBC_CMD(msg=0))

# 2. Heuristic Peak Shaving
heuristic_soc = np.full((T+1, N), capacity/2)
heuristic_grid = np.zeros((T, N))

for t in range(T):
    for i in range(N):
        if 8 <= t < 20:  # Peak hours: discharge first
            d = min(heuristic_soc[t][i], demand[t][i], P_max)
            heuristic_grid[t][i] = demand[t][i] - d
            heuristic_soc[t+1][i] = heuristic_soc[t][i] - d
        else:  # Off-peak: charge if space
            c = min(capacity - heuristic_soc[t][i], P_max)
            heuristic_grid[t][i] = max(0, demand[t][i] - c)
            heuristic_soc[t+1][i] = heuristic_soc[t][i] + c

# Data Analysis and Visualization

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: SOC Heatmap
sns.heatmap(heuristic_soc[:-1, :], annot=False, cmap='YlGnBu', ax=axes[0, 0], cbar_kws={'label': 'State of Charge (kWh)'})
axes[0, 0].set_title('Storage State of Charge (SOC) Distribution')
axes[0, 0].set_xlabel('Stations')
axes[0, 0].set_ylabel('Time')

# Plot 2: Demand Time Series
for i in range(N):
    axes[0, 1].plot(demand[:, i], label=f'Station {i+1}' if i < 3 else "")
axes[0, 1].set_title('Demand Profile Across Stations')
axes[0, 1].set_xlabel('Time (hours)')
axes[0, 1].set_ylabel('Demand (kWh)')
axes[0, 1].legend(title='Stations', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: Price Volatility
axes[1, 0].plot(price, marker='o', color='darkred')
axes[1, 0].set_title('Electricity Price Volatility')
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Price (¥/kWh)')
axes[1, 0].fill_between(range(T), price, where=(price > 0.5), color='lightcoral', alpha=0.3, label='Peak Hours')
axes[1, 0].fill_between(range(T), price, where=(price <= 0.5), color='lightgreen', alpha=0.3, label='Off-Peak')
axes[1, 0].legend()

# Plot 4: Charging/Discharging Behavior
axes[1, 1].plot(np.sum(heuristic_grid, axis=1), label='Total Grid Draw', color='orange')
axes[1, 1].set_title('Grid Load Profile')
axes[1, 1].set_xlabel('Time (hours)')
axes[1, 1].set_ylabel('Grid Power (kW)')
axes[1, 1].legend()

# Prepare data for clustering
df = pd.DataFrame(demand.T, columns=[f'Hour {h}' for h in range(T)])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Determine optimal number of clusters using Silhouette score
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

# Choose optimal K
optimal_k = 2 + np.argmax(silhouette_scores)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Plot 5: Clustering Results
for cluster in range(optimal_k):
    cluster_data = df[clusters == cluster]
    axes[2, 0].plot(cluster_data.mean(axis=0), marker='o', label=f'Cluster {cluster+1}')
axes[2, 0].set_title(f'Demand Pattern Clustering (K={optimal_k})')
axes[2, 0].set_xlabel('Time (hours)')
axes[2, 0].set_ylabel('Normalized Demand')
axes[2, 0].legend(title='Clusters')

# Plot 6: Cost Comparison
cost_opt = pulp.value(pulp.lpSum(grid[t][i]*price[t] for t in range(T) for i in range(N)))
peak_opt = pulp.value(peak)
cost_heuristic = np.sum(heuristic_grid * price[:, np.newaxis])
peak_heuristic = np.max(np.sum(heuristic_grid, axis=1))

axes[2, 1].bar(['Optimization', 'Heuristic'], [cost_opt, cost_heuristic], color=['mediumseagreen', 'cornflowerblue'])
axes[2, 1].set_title('Cost Comparison')
axes[2, 1].set_ylabel('Total Cost (¥)')
axes[2, 1].bar_label(axes[2, 1].containers[0], label_type='edge')

# Adjust layout
plt.tight_layout()

# Print results
print(f"优化结果: 总成本 = {cost_opt:.2f} 元, 峰值 = {peak_opt:.2f} kW")
print(f"启发式结果: 总成本 = {cost_heuristic:.2f} 元, 峰值 = {peak_heuristic:.2f} kW")

# Save the figure
plt.savefig('storage_scheduling_analysis.png', dpi=300)
plt.show()