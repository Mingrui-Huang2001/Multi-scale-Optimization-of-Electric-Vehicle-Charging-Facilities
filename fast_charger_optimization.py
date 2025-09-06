# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import random
import math
from typing import List, Dict, Tuple
import matplotlib
import sys

# 设置中文字体
try:
    if sys.platform == 'win32':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    elif sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'sans-serif'
except:
    print("警告：中文字体设置失败")

class DemandPredictor:
    """充电需求预测模型"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        
    def fit(self, demand_points: List[Tuple[float, float]], demand_values: List[float]):
        """训练需求预测模型"""
        self.demand_points = np.array(demand_points)
        self.demand_values = np.array(demand_values)
        self.kmeans.fit(self.demand_points)
        
    def predict(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """预测指定位置的需求"""
        clusters = self.kmeans.predict(points)
        return np.array([self.demand_values[self.kmeans.labels_ == c].mean() 
                        for c in clusters])

# 定义DEAP creator类(全局只定义一次)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -0.5, -0.1))
creator.create("Individual", list, fitness=creator.FitnessMulti)

class ChargerLayoutOptimizer:
    """快充桩布局优化器"""
    
    def __init__(self, n_chargers=500, max_power=120, min_power=60):
        self.n_chargers = n_chargers
        self.max_power = max_power
        self.min_power = min_power
        self.charger_cost = 50000  # 每个充电桩建设成本(元)
        self.max_grid_load = 50000  # 电网最大允许负荷(kW)
        
    def initialize_population(self, demand_predictor: DemandPredictor, 
                            area_size: Tuple[float, float] = (10, 10)):
        """初始化遗传算法种群"""
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # 定义个体生成函数
        def generate_individual():
            locations = [(random.uniform(0, area_size[0]), 
                         random.uniform(0, area_size[1])) 
                        for _ in range(self.n_chargers)]
            powers = [random.randint(self.min_power, self.max_power) 
                     for _ in range(self.n_chargers)]
            return locations + powers
            
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # 注册遗传算子
        def mate_individuals(ind1, ind2, alpha=0.5):
            """Custom crossover for mixed-type individuals"""
            # Crossover locations (first n_chargers elements)
            for i in range(self.n_chargers):
                if random.random() < 0.7:  # crossover probability
                    # Blend crossover for coordinates
                    ind1[i] = (
                        (1. - alpha) * ind1[i][0] + alpha * ind2[i][0],
                        (1. - alpha) * ind1[i][1] + alpha * ind2[i][1]
                    )
                    ind2[i] = (
                        (1. - alpha) * ind2[i][0] + alpha * ind1[i][0],
                        (1. - alpha) * ind2[i][1] + alpha * ind1[i][1]
                    )
            
            # Crossover powers (remaining elements)
            for i in range(self.n_chargers, len(ind1)):
                if random.random() < 0.7:  # crossover probability
                    # Standard blend crossover
                    ind1[i] = int((1. - alpha) * ind1[i] + alpha * ind2[i])
                    ind2[i] = int((1. - alpha) * ind2[i] + alpha * ind1[i])
            
            return ind1, ind2

        toolbox.register("mate", mate_individuals, alpha=0.5)
        toolbox.register("mutate", self.mutate_individual, indpb=0.1)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", self.evaluate_individual, 
                        demand_predictor=demand_predictor)
                        
        return toolbox
        
    def mutate_individual(self, individual, indpb):
        """变异算子"""
        for i in range(len(individual)):
            if random.random() < indpb:
                if i < self.n_chargers:  # 位置变异
                    individual[i] = (individual[i][0] + random.gauss(0, 0.5),
                                   individual[i][1] + random.gauss(0, 0.5))
                else:  # 功率变异
                    individual[i] = random.randint(self.min_power, self.max_power)
        return individual,
        
    def evaluate_individual(self, individual, demand_predictor):
        """评估个体适应度"""
        locations = individual[:self.n_chargers]
        powers = individual[self.n_chargers:]
        
        # 1. 计算覆盖率(95%需求点10分钟内可达)
        coverage = self.calculate_coverage(locations, demand_predictor.demand_points)
        
        # 2. 计算电网负荷波动
        load_fluctuation = self.calculate_load_fluctuation(locations, powers, demand_predictor)
        
        # 3. 计算总建设成本
        total_cost = sum(powers) * self.charger_cost / self.max_power
        
        return coverage, load_fluctuation, total_cost
        
    def calculate_coverage(self, chargers, demand_points, max_distance=1.5):
        """计算需求点覆盖率(假设1.5单位距离≈10分钟车程)"""
        covered = 0
        for point in demand_points:
            min_dist = min(math.sqrt((point[0]-c[0])**2 + (point[1]-c[1])**2) for c in chargers)
            if min_dist <= max_distance:
                covered += 1
        return covered / len(demand_points)
        
    def calculate_load_fluctuation(self, chargers, powers, demand_predictor):
        """计算电网负荷波动率"""
        # 预测每个充电桩的负荷
        demands = demand_predictor.predict(chargers)
        loads = [d * p for d, p in zip(demands, powers)]
        
        # 计算区域负荷(简单划分为5x5网格)
        grid_loads = np.zeros((5,5))
        area_size = (10, 10)
        cell_size = (area_size[0]/5, area_size[1]/5)
        
        for (x,y), load in zip(chargers, loads):
            i = min(int(x / cell_size[0]), 4)
            j = min(int(y / cell_size[1]), 4)
            grid_loads[i,j] += load
            
        # 计算波动率(标准差/均值)
        return np.std(grid_loads) / np.mean(grid_loads)
        
    def optimize(self, demand_predictor, ngen=50, pop_size=100):
        """运行优化算法"""
        toolbox = self.initialize_population(demand_predictor)
        pop = toolbox.population(n=pop_size)
        
        # 运行NSGA-II算法
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=ngen, 
                          verbose=False)
        
        # 获取Pareto前沿解
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        # 选择最优解(覆盖率>95%,波动率<30%,成本最低)
        feasible = [ind for ind in pareto_front 
                   if ind.fitness.values[0] >= 0.95 
                   and ind.fitness.values[1] <= 0.3]
        
        if feasible:
            best = min(feasible, key=lambda x: x.fitness.values[2])
        else:
            best = max(pareto_front, key=lambda x: x.fitness.values[0])
            
        return best, pareto_front

class GridSafetyAnalyzer:
    """电网安全分析器"""
    
    def __init__(self, base_load=1000):
        self.base_load = base_load  # 电网基础负荷(kW)
        
    def analyze(self, chargers, powers, demands):
        """分析电网安全指标"""
        total_load = sum(d * p for d, p in zip(demands, powers))
        max_cell_load = self._calculate_cell_loads(chargers, powers, demands)
        
        return {
            "total_load": total_load,
            "load_percentage": total_load / self.base_load,
            "max_cell_load": max_cell_load,
            "max_cell_percentage": max_cell_load / self.base_load,
            "fluctuation": np.std([d * p for d, p in zip(demands, powers)]) / total_load
        }
        
    def _calculate_cell_loads(self, chargers, powers, demands):
        """计算网格单元负荷(5x5网格)"""
        grid_loads = np.zeros((5,5))
        for (x,y), p, d in zip(chargers, powers, demands):
            i = min(int(x / 2), 4)  # 假设区域大小为10x10
            j = min(int(y / 2), 4)
            grid_loads[i,j] += d * p
        return np.max(grid_loads)

def generate_sample_data(n_points=1000):
    """生成示例需求点数据"""
    # 生成空间位置(聚类分布)
    centers = [(2,2), (2,8), (8,2), (8,8), (5,5)]
    points = []
    for c in centers:
        points.extend([(c[0] + random.gauss(0, 0.5), 
                       c[1] + random.gauss(0, 0.5)) 
                      for _ in range(n_points//5)])
    
    # 生成需求值(与中心距离相关)
    demands = []
    for x,y in points:
        dist_to_center = min(math.sqrt((x-cx)**2 + (y-cy)**2) for cx,cy in centers)
        demands.append(max(0.1, 1 - dist_to_center/5) + random.gauss(0, 0.1))
        
    return points, demands

def visualize_results(best_individual, demand_points, demand_predictor):
    """可视化优化结果"""
    plt.figure(figsize=(15, 5))
    
    # 1. 布局图
    plt.subplot(1, 3, 1)
    locations = best_individual[:500]
    powers = best_individual[500:]
    
    # 绘制需求点
    clusters = demand_predictor.kmeans.predict(demand_points)
    for c in range(demand_predictor.n_clusters):
        cluster_points = np.array(demand_points)[clusters == c]
        plt.scatter(cluster_points[:,0], cluster_points[:,1], s=5, 
                   label=f'需求簇{c+1}')
    
    # 绘制充电桩
    for (x,y), p in zip(locations, powers):
        plt.scatter(x, y, c='red', marker='^', s=p/2, alpha=0.7)
    
    plt.title('快充桩布局优化结果')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    
    # 2. 功率分布
    plt.subplot(1, 3, 2)
    plt.hist(powers, bins=10, color='skyblue', edgecolor='black')
    plt.title('充电桩功率分布')
    plt.xlabel('功率(kW)')
    plt.ylabel('数量')
    
    # 3. 覆盖范围
    plt.subplot(1, 3, 3)
    covered = []
    for point in demand_points:
        min_dist = min(math.sqrt((point[0]-c[0])**2 + (point[1]-c[1])**2) for c in locations)
        covered.append(min_dist <= 1.5)
    
    plt.scatter([p[0] for p in demand_points], [p[1] for p in demand_points],
               c=['green' if c else 'red' for c in covered], s=2)
    plt.title('需求点覆盖情况(绿色=已覆盖)')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    
    plt.tight_layout()
    plt.savefig('charger_layout.png')
    print("已生成布局优化图: charger_layout.png")

def sensitivity_analysis(optimizer, demand_predictor, n_runs=10):
    """灵敏度分析"""
    results = []
    for _ in range(n_runs):
        best, _ = optimizer.optimize(demand_predictor, ngen=30)
        coverage, fluctuation, cost = best.fitness.values
        results.append({
            "coverage": coverage,
            "fluctuation": fluctuation,
            "cost": cost
        })
    
    # 计算灵敏度指标
    coverage_var = np.std([r["coverage"] for r in results])
    fluctuation_var = np.std([r["fluctuation"] for r in results])
    cost_var = np.std([r["cost"] for r in results])
    
    print("\n灵敏度分析结果:")
    print(f"覆盖率标准差: {coverage_var:.4f}")
    print(f"波动率标准差: {fluctuation_var:.4f}")
    print(f"成本标准差: {cost_var:.4f}")
    
    return results

if __name__ == "__main__":
    # 设置环境变量避免已知问题
    import os
    os.environ['OMP_NUM_THREADS'] = '1'  # 避免K-means内存泄漏
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # 限制CPU核心使用
    
    # 1. 生成示例数据
    print("正在生成示例数据...")
    demand_points, demand_values = generate_sample_data(n_points=500)  # 减少数据点数量
    
    # 2. 训练需求预测模型
    print("训练需求预测模型...")
    demand_predictor = DemandPredictor(n_clusters=3)  # 减少聚类数量
    demand_predictor.fit(demand_points, demand_values)
    
    # 3. 运行布局优化
    print("运行布局优化(可能需要几分钟)...")
    optimizer = ChargerLayoutOptimizer(n_chargers=500)
    
    # 添加进度回调
    def print_progress(gen, pop):
        best = tools.selBest(pop, 1)[0]
        print(f"第{gen}代: 覆盖率={best.fitness.values[0]:.2%} "
              f"波动率={best.fitness.values[1]:.2%} "
              f"成本={best.fitness.values[2]/1e6:.2f}百万")
        return False  # 不停止
    
    # 使用更小的种群和代数
    best_individual, pareto_front = optimizer.optimize(
        demand_predictor, 
        ngen=30,  # 减少代数
        pop_size=50  # 减少种群大小
    )
    
    # 4. 分析结果
    coverage, fluctuation, cost = best_individual.fitness.values
    print("\n优化结果:")
    print(f"需求覆盖率: {coverage:.2%}")
    print(f"电网波动率: {fluctuation:.2%}")
    print(f"总建设成本: {cost/1e6:.2f} 百万元")
    
    # 5. 电网安全分析
    print("\n电网安全分析:")
    locations = best_individual[:500]
    powers = best_individual[500:]
    demands = demand_predictor.predict(locations)
    
    grid_analyzer = GridSafetyAnalyzer(base_load=5000)
    safety_report = grid_analyzer.analyze(locations, powers, demands)
    print(f"总负荷: {safety_report['total_load']/1000:.1f} MW")
    print(f"最大区域负荷占比: {safety_report['max_cell_percentage']:.1%}")
    
    # 6. 可视化结果
    visualize_results(best_individual, demand_points, demand_predictor)
    
    # 7. 灵敏度分析
    sensitivity_analysis(optimizer, demand_predictor)