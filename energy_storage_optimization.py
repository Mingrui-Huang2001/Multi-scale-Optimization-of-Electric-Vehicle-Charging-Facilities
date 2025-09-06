# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import random
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

class EnergyStorageOptimizer:
    """储能型充电桩动态调度优化模型"""
    
    def __init__(self, n_chargers=100, storage_capacity=50):
        """
        初始化模型
        参数:
            n_chargers: 充电桩数量
            storage_capacity: 单桩储能容量(kWh)
        """
        self.n_chargers = n_chargers
        self.storage_capacity = storage_capacity
        self.peak_power = 120  # 快充桩峰值功率(kW)
        self.base_load = 5000  # 电网基础负荷(kW)
        
        # 电价时段划分 (小时)
        self.price_periods = {
            'valley': (0, 8, 0.3),   # 谷电时段: 0-8点, 0.3元/kWh
            'flat': (8, 16, 0.5),     # 平电时段: 8-16点, 0.5元/kWh
            'peak': (16, 24, 0.8)     # 峰电时段: 16-24点, 0.8元/kWh
        }
        
        # 初始化历史数据
        self.original_cost = None
        self.original_fluctuation = None
    
    def simulate_demand(self, hours=24):
        """模拟24小时充电需求"""
        demand = []
        for hour in range(hours):
            if self._is_peak_time(hour):
                # 峰时段需求较高
                demand.append(random.randint(30, 50) * self.n_chargers / 100)
            else:
                # 其他时段需求较低
                demand.append(random.randint(10, 30) * self.n_chargers / 100)
        return demand
    
    def _is_peak_time(self, hour):
        """判断是否为峰时段"""
        return self.price_periods['peak'][0] <= hour < self.price_periods['peak'][1]
    
    def optimize_schedule(self, demand: List[float]):
        """优化储能充放电计划"""
        time_slots = len(demand)
        
        # 定义优化变量: 每个时段的储能充放电功率
        # 正数表示充电,负数表示放电
        x0 = np.zeros(time_slots)
        
        # 定义边界 (考虑储能功率限制)
        bounds = [(-self.peak_power * 0.8, self.peak_power * 0.5) for _ in range(time_slots)]
        
        # 定义约束条件
        constraints = [
            # 储能容量约束
            {'type': 'ineq', 'fun': lambda x: self.storage_capacity - np.cumsum(x)[-1]},
            {'type': 'ineq', 'fun': lambda x: self.storage_capacity + np.cumsum(x)[-1]},
            
            # 电网负荷波动约束 (不超过30%)
            {'type': 'ineq', 'fun': lambda x: 
             0.3 * self.base_load - np.std([d * self.peak_power + p for d, p in zip(demand, x)])}
        ]
        
        # 优化目标: 最小化总成本
        def cost_function(x):
            total_cost = 0
            storage = 0  # 当前储能电量
            for i in range(time_slots):
                hour = i % 24
                price = self._get_electricity_price(hour)
                
                # 计算净功率
                net_power = demand[i] * self.peak_power + x[i]
                
                # 计算成本
                total_cost += net_power * price
                
                # 更新储能状态
                storage += x[i]
                storage = max(0, min(self.storage_capacity, storage))
            
            return total_cost
        
        # 记录原始成本
        self.original_cost = sum(d * self.peak_power * self._get_electricity_price(i % 24) 
                                for i, d in enumerate(demand))
        
        # 记录原始负荷波动
        original_loads = [d * self.peak_power for d in demand]
        peak_loads = [load for i, load in enumerate(original_loads) 
                     if self._is_peak_time(i % 24)]
        self.original_fluctuation = np.std(peak_loads) / np.mean(peak_loads)
        
        # 添加进度回调
        def callback(xk):
            current_cost = cost_function(xk)
            print(f"当前迭代: 成本={current_cost:.2f}元", end='\r')
        
        # 运行优化
        print("\n开始优化...")
        try:
            res = minimize(
                cost_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-4, 'disp': True},
                callback=callback
            )
            print("\n优化完成!")
        except Exception as e:
            print(f"\n优化过程中出错: {str(e)}")
            return np.zeros_like(x0)
        
        return res.x
    def optimize_schedule(self, demand: List[float]):
        """动态调度策略核心算法"""
        # 定义电价时段 (谷电0-8点，峰电16-24点)
        valley_hours = list(range(self.price_periods['valley'][0], self.price_periods['valley'][1]))
        peak_hours = list(range(self.price_periods['peak'][0], self.price_periods['peak'][1]))
        
        schedule = np.zeros(len(demand))
        storage = 0
        
        for t in range(len(demand)):
            # 谷电时段充电
            if t in valley_hours:
                charge_power = min(self.storage_capacity - storage, self.peak_power * 0.5)
                schedule[t] = charge_power
                storage += charge_power
            # 峰电时段放电
            elif t in peak_hours and storage > 0:
                discharge_power = min(demand[t] * self.peak_power, storage, self.peak_power * 0.8)
                schedule[t] = -discharge_power
                storage -= discharge_power
                
        return schedule
    
    def _get_electricity_price(self, hour):
        """获取当前时段电价"""
        for period, (start, end, price) in self.price_periods.items():
            if start <= hour < end:
                return price
        return 0.5  # 默认价格
    
    def evaluate(self, demand: List[float], schedule: List[float]):
        """评估优化效果"""
        # 计算优化后成本
        optimized_cost = 0
        storage = 0
        loads = []
        for i in range(len(demand)):
            hour = i % 24
            price = self._get_electricity_price(hour)
            
            # 计算净功率和成本
            net_power = demand[i] * self.peak_power + schedule[i]
            optimized_cost += net_power * price
            
            # 记录负荷
            loads.append(net_power)
            
            # 更新储能状态
            storage += schedule[i]
            storage = max(0, min(self.storage_capacity, storage))
        
        # 计算成本降低比例
        cost_reduction = (self.original_cost - optimized_cost) / self.original_cost * 100
        
        # 计算负荷波动
        peak_loads = [load for i, load in enumerate(loads) 
                     if self._is_peak_time(i % 24)]
        optimized_fluctuation = np.std(peak_loads) / np.mean(peak_loads)
        fluctuation_reduction = (self.original_fluctuation - optimized_fluctuation) / self.original_fluctuation * 100
        
        # 打印结果
        print("\n优化结果:")
        print(f"原始总成本: {self.original_cost:.2f} 元")
        print(f"优化后总成本: {optimized_cost:.2f} 元")
        print(f"成本降低: {cost_reduction:.2f}%")
        print(f"原始峰时段负荷波动率: {self.original_fluctuation:.2%}")
        print(f"优化后峰时段负荷波动率: {optimized_fluctuation:.2%}")
        print(f"负荷波动降低: {fluctuation_reduction:.2f}%")
        
        # 可视化结果
        self.visualize(demand, loads, schedule)
        
        return {
            'cost_reduction': cost_reduction,
            'fluctuation_reduction': fluctuation_reduction,
            'optimized_schedule': schedule
        }
    def evaluate(self, demand: List[float], schedule: List[float]):
        """包含双重约束的评估函数"""
        # 负荷波动计算
        original_load = np.array(demand) * self.peak_power
        adjusted_load = original_load + np.where(np.array(schedule) < 0, np.array(schedule), 0)
        
        # 峰时段波动率下降指标
        peak_mask = np.isin(range(24), list(range(self.price_periods['peak'][0], self.price_periods['peak'][1])))
        orig_std = np.std(original_load[peak_mask])
        new_std = np.std(adjusted_load[peak_mask])
        
        # 成本计算（仅计算放电时段替代量）
        valley_hours = list(range(self.price_periods['valley'][0], self.price_periods['valley'][1]))
        peak_hours = list(range(self.price_periods['peak'][0], self.price_periods['peak'][1]))
        
        valley_energy = sum(schedule[t] for t in valley_hours if schedule[t] > 0)
        peak_energy = sum(-schedule[t] for t in peak_hours if schedule[t] < 0)
        orig_cost = peak_energy * self.price_periods['peak'][2]
        new_cost = valley_energy * self.price_periods['valley'][2]
        
        cost_reduction = (orig_cost - new_cost)/orig_cost * 100 if orig_cost > 0 else 0
        fluctuation_reduction = (orig_std - new_std)/orig_std * 100 if orig_std > 0 else 0
        
        # 打印结果
        print("\n优化结果:")
        print(f"原始峰时段成本: {orig_cost:.2f} 元")
        print(f"优化后谷时段成本: {new_cost:.2f} 元")
        print(f"成本降低: {cost_reduction:.2f}%")
        print(f"原始峰时段负荷波动率: {orig_std:.2f}")
        print(f"优化后峰时段负荷波动率: {new_std:.2f}")
        print(f"负荷波动降低: {fluctuation_reduction:.2f}%")
        
        # 可视化结果
        self.visualize(demand
    
    def visualize(self, demand: List[float], loads: List[float], schedule: List[float]):
        # 在第一个子图中新增充放电效率标注
        plt.subplot(3, 1, 1)
        plt.text(22, max(loads)*0.9, f'充电效率: {self.charging_efficiency*100}%\n放电效率: {self.discharging_efficiency*100}%',
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # 在第二个子图中新增电价标注
        plt.subplot(3, 1, 2)
        for hour in range(24):
            price = self._get_electricity_price(hour)
            plt.text(hour, max(schedule)*0.8, f'{price}元', rotation=90, ha='center', fontsize=8)
        
        # 原有可视化代码保持不变...

        plt.tight_layout()
        plt.savefig('storage_optimization_results.png')
        print("\n已生成优化结果图: storage_optimization_results.png")

if __name__ == "__main__":
    # 初始化优化器
    optimizer = EnergyStorageOptimizer(n_chargers=100, storage_capacity=50)
    
    # 模拟24小时需求
    demand = optimizer.simulate_demand(hours=24)
    
    # 运行优化
    schedule = optimizer.optimize_schedule(demand)
    
    # 评估结果
    results = optimizer.evaluate(demand, schedule)
    
    # 检查是否达到目标
    if results['cost_reduction'] >= 20 and results['fluctuation_reduction'] >= 50:
        print("\n优化目标已达成!")
    else:
        print("\n优化目标未完全达成，建议调整参数或优化算法")