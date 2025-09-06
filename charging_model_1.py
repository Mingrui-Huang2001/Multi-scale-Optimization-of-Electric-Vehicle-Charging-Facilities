# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from scipy.optimize import minimize
import matplotlib
import sys
from typing import Dict, List, Tuple

# 设置中文字体，支持多平台
try:
    if sys.platform == 'win32':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    elif sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 强制使用支持中文的字体
    matplotlib.rcParams['font.family'] = 'sans-serif'
except:
    print("警告：中文字体设置失败，图表可能显示异常")

# 确保标准输出支持中文
try:
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    print("警告：标准输出编码设置失败")

class ChargingModel:
    def __init__(self, n_chargers: int = 1000, power: float = 7.0):
        """
        优化后的充电桩共享模型
        参数:
            n_chargers: 充电桩数量(默认1000)
            power: 单个充电桩功率(kW, 默认7kW)
        """
        self.n_chargers = n_chargers
        self.power = power
        self.shared_ratio = 0.5  # 初始共享比例
        self.avg_charging_energy = 30  # 平均充电量(kWh)
        self.optimization_history = []  # 记录优化过程
        self.sharing_strategy = "time_sharing"  # 默认共享策略
        self.time_slots = {
            "morning": (8, 12),
            "afternoon": (12, 18),
            "evening": (18, 20)
        }
        
    def set_user_behavior(self, lambda_night: float = 0.3, lambda_day: float = 0.1,
                        daily_pattern: Dict[str, float] = None):
        """
        设置用户充电行为参数
        参数:
            lambda_night: 夜间充电需求泊松分布参数(自有时段)
            lambda_day: 日间充电需求泊松分布参数(共享时段)
            daily_pattern: 日间需求模式字典 {时段: 需求系数}
        """
        self.lambda_night = lambda_night
        self.lambda_day = lambda_day
        self.daily_pattern = daily_pattern or {
            'morning': 1.2,   # 8:00-12:00
            'afternoon': 0.8, # 12:00-18:00
            'evening': 1.0    # 18:00-20:00
        }
        
    def simulate_demand(self, hours=12, shared=False):
        """
        模拟充电需求
        :param hours: 模拟时长(小时)
        :param shared: 是否为共享时段
        :return: 每小时需求列表
        """
        lambda_param = self.lambda_day if shared else self.lambda_night
        return poisson.rvs(lambda_param * self.n_chargers, size=hours)
        
    def apply_sharing_strategy(self, demand, current_hour):
        """
        应用共享策略，返回可用充电桩数量
        :param demand: 当前小时需求
        :param current_hour: 当前小时(0-23)
        :return: 可用充电桩数量
        """
        if self.sharing_strategy == "fixed_ratio":
            return int(self.n_chargers * self.shared_ratio)
            
        elif self.sharing_strategy == "time_sharing":
            for slot, (start, end) in self.time_slots.items():
                if start <= current_hour < end:
                    return int(self.n_chargers * self.shared_ratio)
            return 0
            
        elif self.sharing_strategy == "dynamic":
            # 动态策略基于需求预测调整共享比例
            predicted_demand = self.predict_demand(current_hour)
            return min(int(self.n_chargers * 0.8), predicted_demand)
            
        else:
            return 0
            
    def predict_demand(self, hour):
        """
        简单需求预测模型
        :param hour: 当前小时(0-23)
        :return: 预测需求数量
        """
        # 基于时间段的简单线性预测
        if 8 <= hour < 12:  # 上午
            return self.n_chargers * 0.3
        elif 12 <= hour < 18:  # 下午
            return self.n_chargers * 0.5
        else:  # 晚上
            return self.n_chargers * 0.4
        
    def calculate_wait_time(self, demand, available_chargers):
        """
        计算平均等待时间
        :param demand: 充电需求数量
        :param available_chargers: 可用充电桩数量
        :return: 平均等待时间(小时)
        """
        excess = np.maximum(demand - available_chargers, 0)
        return np.mean(excess / available_chargers) if available_chargers > 0 else float('inf')
        
    def calculate_grid_load(self, charging_count):
        """
        计算电网负荷
        :param charging_count: 正在使用的充电桩数量
        :return: 总负荷(kW)
        """
        return charging_count * self.power
        
    def calculate_load_balance(self, load_series):
        """
        计算电网负荷均衡性指标
        :param load_series: 负荷时间序列(kW)
        :return: 字典包含各种均衡性指标
        """
        if not load_series:
            return {}
            
        load_arr = np.array(load_series)
        max_load = np.max(load_arr)
        min_load = np.min(load_arr)
        avg_load = np.mean(load_arr)
        
        return {
            "peak_valley_diff": max_load - min_load,  # 峰谷差
            "fluctuation_rate": np.std(load_arr) / avg_load,  # 波动率
            "load_factor": avg_load / max_load,  # 负荷率
            "max_load": max_load,
            "min_load": min_load,
            "avg_load": avg_load
        }

# 示例使用
if __name__ == "__main__":
    # 初始化模型
    model = ChargingModel(n_chargers=1000, power=7)
    model.set_user_behavior(lambda_night=0.3, lambda_day=0.2)
    
    # 测试三种共享策略
    strategies = ["fixed_ratio", "time_sharing", "dynamic"]
    strategy_names = ["固定比例", "分时共享", "动态调整"]
    results = []
    
    for strategy in strategies:
        model.sharing_strategy = strategy
        print(f"\n正在测试策略: {strategy_names[strategies.index(strategy)]}")
        
        # 24小时仿真
        hourly_demand = []
        hourly_load = []
        hourly_available = []
        
        for hour in range(24):
            # 判断当前时段
            if 20 <= hour or hour < 8:  # 夜间自有充电时段
                demand = model.simulate_demand(hours=1, shared=False)[0]
                available = model.n_chargers
            else:  # 日间共享充电时段
                demand = model.simulate_demand(hours=1, shared=True)[0]
                available = model.apply_sharing_strategy(demand, hour)
            
            # 计算实际使用量和电网负荷
            usage = min(demand, available)
            load = model.calculate_grid_load(usage)
            
            # 记录数据
            hourly_demand.append(demand)
            hourly_available.append(available)
            hourly_load.append(load)
        
        # 计算关键指标
        night_load = hourly_load[20:] + hourly_load[:8]
        day_load = hourly_load[8:20]
        
        night_balance = model.calculate_load_balance(night_load)
        day_balance = model.calculate_load_balance(day_load)
        
        # 计算等待时间
        wait_times = []
        for h in range(24):
            if 8 <= h < 20:  # 只在共享时段计算等待时间
                wait = max(0, hourly_demand[h] - hourly_available[h]) / hourly_available[h] if hourly_available[h] > 0 else float('inf')
                wait_times.append(wait)
        avg_wait = np.mean(wait_times)
        
        # 存储结果
        results.append({
            "strategy": strategy_names[strategies.index(strategy)],
            "hourly_load": hourly_load,
            "hourly_demand": hourly_demand,
            "avg_wait": avg_wait,
            "night_balance": night_balance,
            "day_balance": day_balance
        })
        
        # 输出当前策略结果
        print(f"日间平均等待时间: {avg_wait:.2f}小时")
        print("夜间电网负荷指标:")
        for k, v in night_balance.items():
            print(f"{k}: {v:.2f}")
        print("日间电网负荷指标:")
        for k, v in day_balance.items():
            print(f"{k}: {v:.2f}")
    
    # 生成比较图表
    plt.figure(figsize=(15, 10))
    
    # 1. 负荷曲线比较
    plt.subplot(2, 1, 1)
    for res in results:
        plt.plot(range(24), res["hourly_load"], label=f"{res['strategy']}负荷")
    plt.plot(range(24), results[0]["hourly_demand"], 'k--', label='充电需求')
    plt.axvspan(8, 20, alpha=0.1, color='green', label='共享时段')
    plt.xlabel('时间(小时)')
    plt.ylabel('负荷(kW)')
    plt.title('不同共享策略下的电网负荷比较')
    plt.legend()
    plt.grid()
    
    # 2. 性能指标比较
    plt.subplot(2, 1, 2)
    metrics = ["avg_wait", ("day_balance", "max_load"), ("night_balance", "peak_valley_diff")]
    metric_names = ["平均等待时间(小时)", "日间最大负荷(kW)", "夜间峰谷差(kW)"]
    
    x = np.arange(len(metric_names))
    width = 0.25
    for i, res in enumerate(results):
        values = []
        for metric in metrics:
            if isinstance(metric, tuple):
                values.append(res[metric[0]][metric[1]])
            else:
                values.append(res[metric])
        plt.bar(x + i*width, values, width, label=res["strategy"])
    
    plt.xlabel('性能指标')
    plt.ylabel('数值')
    plt.title('不同共享策略性能比较')
    plt.xticks(x + width, metric_names)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    print("\n已生成策略比较图: strategy_comparison.png")

    # 新增数据分析部分
    from sklearn.cluster import KMeans
    import seaborn as sns

    # 准备聚类数据
    cluster_data = []
    for res in results:
        cluster_data.append(res["hourly_load"])
    cluster_data = np.array(cluster_data).T  # 转置为(24小时×3策略)

    # K-means聚类分析 (设置环境变量避免Windows上的内存泄漏)
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(cluster_data)
    labels = kmeans.labels_

    # 生成热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_data, cmap="YlGnBu", 
                xticklabels=[res["strategy"] for res in results],
                yticklabels=range(24))
    plt.title('不同策略24小时负荷热力图')
    plt.xlabel('共享策略')
    plt.ylabel('时间(小时)')
    plt.savefig('load_heatmap.png')
    print("已生成负荷热力图: load_heatmap.png")

    # 生成聚类结果图
    plt.figure(figsize=(10, 6))
    for i in range(3):  # 3个簇
        cluster_hours = np.where(labels == i)[0]
        for hour in cluster_hours:
            plt.scatter(hour, cluster_data[hour, 0], color=f'C{i}', label=f'簇{i+1}' if hour == cluster_hours[0] else "")
    plt.xlabel('时间(小时)')
    plt.ylabel('负荷(kW)')
    plt.title('K-means聚类结果(按负荷模式)')
    plt.legend()
    plt.grid()
    plt.savefig('load_clusters.png')
    print("已生成负荷聚类图: load_clusters.png")

    # 输出聚类分析结果
    print("\n聚类分析结果:")
    for i in range(3):
        cluster_hours = np.where(labels == i)[0]
        avg_load = np.mean(cluster_data[cluster_hours, :])
        print(f"簇{i+1}: 包含时段 {list(cluster_hours)}")
        print(f"    平均负荷: {avg_load:.2f} kW")
        print(f"    主要特征: {'高峰' if avg_load > 1800 else '平峰' if avg_load > 1200 else '低谷'}时段")