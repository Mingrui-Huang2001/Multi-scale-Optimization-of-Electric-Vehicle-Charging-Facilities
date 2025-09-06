# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from typing import Dict, List, Tuple
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

class ChargingOptimizer:
    """充电桩共享策略优化模型"""
    
    def __init__(self, n_chargers=1000, power=7.0):
        """
        初始化模型
        参数:
            n_chargers: 充电桩总数
            power: 单桩功率(kW)
        """
        self.n_chargers = n_chargers
        self.power = power
        self.time_slots = {
            'private': (0, 8),    # 自有时段(0-8点)
            'shared': (8, 20),    # 共享时段(8-20点)
            'private2': (20, 24)  # 自有时段(20-24点)
        }
        self.shared_ratio = 0.5  # 初始共享比例
        self.avg_charging_time = 2  # 平均充电时长(小时)
        
    def set_demand_params(self, private_lambda=0.3, shared_lambda=0.5):
        """
        设置充电需求参数
        参数:
            private_lambda: 自有时段泊松分布参数
            shared_lambda: 共享时段泊松分布参数
        """
        self.private_lambda = private_lambda
        self.shared_lambda = shared_lambda
        
    def simulate_demand(self, hours=24):
        """模拟24小时充电需求"""
        demand = []
        for hour in range(hours):
            if self._is_private_time(hour):
                # 自有时段需求
                demand.append(poisson.rvs(self.private_lambda * self.n_chargers))
            else:
                # 共享时段需求
                demand.append(poisson.rvs(self.shared_lambda * self.n_chargers))
        return demand
        
    def _is_private_time(self, hour):
        """判断是否为自有时段"""
        return (hour < self.time_slots['private'][1]) or (hour >= self.time_slots['private2'][0])
        
    def apply_sharing_strategy(self, demand, strategy='fixed'):
        """
        应用共享策略
        参数:
            demand: 每小时需求列表
            strategy: 策略类型(fixed/time/dynamic)
        返回:
            可用充电桩列表
        """
        available = []
        for hour, d in enumerate(demand):
            if self._is_private_time(hour):
                # 自有时段全部可用
                available.append(self.n_chargers)
            else:
                # 共享时段按策略分配
                if strategy == 'fixed':
                    # 固定比例策略
                    available.append(int(self.n_chargers * self.shared_ratio))
                elif strategy == 'time':
                    # 分时策略(早晚高峰不同比例)
                    if 8 <= hour < 12 or 18 <= hour < 20:  # 高峰时段
                        available.append(int(self.n_chargers * 0.7))
                    else:  # 平峰时段
                        available.append(int(self.n_chargers * 0.4))
                elif strategy == 'dynamic':
                    # 动态策略(基于需求预测)
                    predicted = self.predict_demand(hour)
                    ratio = min(0.8, predicted / self.n_chargers)
                    available.append(int(self.n_chargers * ratio))
        return available
        
    def predict_demand(self, hour):
        """简单需求预测模型"""
        # 基于历史数据的简单预测
        if 8 <= hour < 12:  # 早高峰
            return self.n_chargers * 0.6
        elif 12 <= hour < 18:  # 日间平峰
            return self.n_chargers * 0.4
        else:  # 晚高峰
            return self.n_chargers * 0.7
            
    def evaluate_performance(self, demand, available):
        """
        评估策略性能
        返回:
            dict: 包含各项指标
        """
        # 计算实际使用量
        usage = [min(d, a) for d, a in zip(demand, available)]
        
        # 计算电网负荷
        load = [u * self.power for u in usage]
        
        # 计算等待时间
        wait_times = []
        for d, a in zip(demand, available):
            if a > 0:
                wait = max(0, d - a) / a * self.avg_charging_time
            else:
                wait = float('inf')
            wait_times.append(wait)
        avg_wait = np.mean(wait_times)
        
        # 计算负荷均衡指标
        shared_load = [l for h, l in enumerate(load) 
                      if not self._is_private_time(h)]
        if shared_load:
            load_std = np.std(shared_load)
            load_avg = np.mean(shared_load)
            fluctuation = load_std / load_avg
        else:
            fluctuation = 0
            
        return {
            'avg_wait': avg_wait,
            'fluctuation': fluctuation,
            'max_load': max(load),
            'min_load': min(load),
            'load_series': load
        }
        
    def optimize_shared_ratio(self, target_wait=0.5):
        """
        优化共享比例
        参数:
            target_wait: 目标平均等待时间(小时)
        返回:
            最优共享比例
        """
        # 简单二分法搜索最优比例
        low, high = 0.1, 0.9
        for _ in range(10):
            mid = (low + high) / 2
            self.shared_ratio = mid
            demand = self.simulate_demand()
            available = self.apply_sharing_strategy(demand, 'fixed')
            perf = self.evaluate_performance(demand, available)
            if perf['avg_wait'] > target_wait:
                low = mid
            else:
                high = mid
        return (low + high) / 2

def visualize_results(results):
    """可视化不同策略结果"""
    plt.figure(figsize=(15, 8))
    
    # 负荷曲线
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(res['load_series'], label=name)
    plt.title('不同策略下的电网负荷')
    plt.xlabel('时间(小时)')
    plt.ylabel('负荷(kW)')
    plt.legend()
    
    # 等待时间
    plt.subplot(2, 2, 2)
    wait_times = [res['avg_wait'] for res in results.values()]
    plt.bar(results.keys(), wait_times)
    plt.title('平均等待时间比较')
    plt.ylabel('等待时间(小时)')
    
    # 负荷波动
    plt.subplot(2, 2, 3)
    fluctuations = [res['fluctuation'] for res in results.values()]
    plt.bar(results.keys(), fluctuations)
    plt.title('负荷波动率比较')
    plt.ylabel('波动率')
    
    # 峰谷差
    plt.subplot(2, 2, 4)
    peak_valley = [res['max_load'] - res['min_load'] for res in results.values()]
    plt.bar(results.keys(), peak_valley)
    plt.title('负荷峰谷差比较')
    plt.ylabel('峰谷差(kW)')
    
    plt.tight_layout()
    plt.savefig('charging_strategy_comparison.png')
    print("已生成策略比较图: charging_strategy_comparison.png")

if __name__ == "__main__":
    # 初始化优化器
    optimizer = ChargingOptimizer(n_chargers=1000, power=7)
    optimizer.set_demand_params(private_lambda=0.2, shared_lambda=0.4)
    
    # 优化共享比例
    optimal_ratio = optimizer.optimize_shared_ratio(target_wait=0.5)
    print(f"最优共享比例: {optimal_ratio:.2%}")
    
    # 测试不同策略
    strategies = ['fixed', 'time', 'dynamic']
    results = {}
    
    for strategy in strategies:
        demand = optimizer.simulate_demand()
        available = optimizer.apply_sharing_strategy(demand, strategy)
        perf = optimizer.evaluate_performance(demand, available)
        results[strategy] = perf
        print(f"\n策略 '{strategy}' 性能:")
        print(f"平均等待时间: {perf['avg_wait']:.2f}小时")
        print(f"负荷波动率: {perf['fluctuation']:.2%}")
        print(f"最大负荷: {perf['max_load']:.2f}kW")
    
    # 可视化结果
    visualize_results(results)