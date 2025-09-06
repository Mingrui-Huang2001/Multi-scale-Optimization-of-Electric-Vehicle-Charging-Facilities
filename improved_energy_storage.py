# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as minimize_moo
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
    """改进版储能型充电桩动态调度优化模型"""
    
    def __init__(self, n_chargers=100, storage_capacity=100):  # 增加储能容量
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
        self.max_charge_power = 80  # 最大充电功率(kW)
        self.max_discharge_power = 100  # 最大放电功率(kW)
        
        # 电价时段划分 (小时)
        self.price_periods = {
            'valley': (0, 8, 0.3),   # 谷电时段: 0-8点, 0.3元/kWh
            'flat': (8, 16, 0.5),     # 平电时段: 8-16点, 0.5元/kWh
            'peak': (16, 24, 0.8)     # 峰电时段: 16-24点, 0.8元/kWh
        }
        
        # 初始化历史数据
        self.original_cost = None
        self.original_fluctuation = None
    
    def simulate_demand(self, days=1):
        """模拟充电需求(默认1天24小时)"""
        demand = []
        for _ in range(days):
            for hour in range(24):
                if self._is_peak_time(hour):
                    # 峰时段需求较高
                    demand.append(random.randint(40, 60) * self.n_chargers / 100)
                else:
                    # 其他时段需求较低
                    demand.append(random.randint(15, 35) * self.n_chargers / 100)
        return np.array(demand)
    
    def _is_peak_time(self, hour):
        """判断是否为峰时段"""
        return self.price_periods['peak'][0] <= hour < self.price_periods['peak'][1]
    
    def optimize_schedule(self, demand: List[float]):
        """多目标优化储能充放电计划"""
        time_slots = len(demand)
        print(f"时间槽数量: {time_slots}")  # 调试信息
        
        # 定义优化问题
        class StorageProblem(Problem):
            def __init__(self, parent):
                super().__init__(
                    n_var=time_slots,
                    n_obj=2,  # 两个目标：最小化成本，最小化波动
                    n_constr=3,  # 三个约束
                    xl=-parent.max_discharge_power,
                    xu=parent.max_charge_power
                )
                self.parent = parent
            
            def _evaluate(self, x, out, *args, **kwargs):
                # 处理批量输入 (x可能是2D数组)
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                
                n_individuals = x.shape[0]
                total_costs = np.zeros(n_individuals)
                fluctuations = np.zeros(n_individuals)
                g1 = np.zeros(n_individuals)
                g2 = np.zeros(n_individuals)
                g3 = np.zeros(n_individuals)
                
                for i in range(n_individuals):
                    # 向量化计算
                    hours = np.arange(time_slots) % 24
                    prices = np.array([self.parent._get_electricity_price(h) for h in hours])
                    
                    # 计算净功率和成本
                    net_powers = demand * self.parent.peak_power + x[i]
                    costs = net_powers * prices
                    total_costs[i] = np.sum(costs)
                    
                    # 计算储能状态
                    storage_states = np.cumsum(x[i])
                    storage_states = np.clip(storage_states, 0, self.parent.storage_capacity)
                    
                    # 计算峰时段负荷波动
                    peak_mask = np.array([self.parent._is_peak_time(h) for h in hours])
                    peak_loads = net_powers[peak_mask]
                    if len(peak_loads) > 0:
                        fluctuations[i] = np.std(peak_loads) / np.mean(peak_loads)
                    
                    # 计算约束
                    g1[i] = abs(storage_states[-1]) - self.parent.storage_capacity
                    g2[i] = np.std(net_powers) / np.mean(net_powers) - 0.3
                    g3[i] = (self.parent.original_cost - total_costs[i]) / self.parent.original_cost - 0.2
                
                out["F"] = np.column_stack([total_costs, fluctuations * 1000])
                out["G"] = np.column_stack([g1, g2, g3])
        
        # 记录原始数据
        self.original_cost = sum(d * self.peak_power * self._get_electricity_price(i % 24) 
                               for i, d in enumerate(demand))
        original_loads = [d * self.peak_power for d in demand]
        peak_loads = [load for i, load in enumerate(original_loads) 
                     if self._is_peak_time(i % 24)]
        self.original_fluctuation = np.std(peak_loads) / np.mean(peak_loads)
        
        # 运行多目标优化
        print("配置优化算法...", flush=True)
        problem = StorageProblem(self)
        algorithm = NSGA2(
            pop_size=100,  # 增加种群大小
            eliminate_duplicates=True
        )
        
        print("开始优化(最多10代)...", flush=True)
        res = minimize_moo(
            problem, 
            algorithm, 
            ('n_gen', 10),  # 增加代数
            verbose=True,
            callback=lambda algorithm: print(f"已完成 {algorithm.n_gen}/10 代", flush=True)
        )
        print("优化完成!", flush=True)
        
        print(f"优化结果类型: {type(res)}")  # 调试信息
        if res is None or res.opt is None or len(res.opt) == 0:
            print("警告: 优化结果中没有可行解")
            # 提供一个默认的解决方案（全零调度）
            return np.zeros(time_slots)
            
        try:
            # 选择最优解(权衡成本和波动)
            best_idx = np.argmin([sol.F[0] * 0.7 + sol.F[1] * 0.3 for sol in res.opt])
            print(f"找到最优解，成本: {res.opt[best_idx].F[0]:.2f}，波动: {res.opt[best_idx].F[1]:.2f}")
            return res.opt[best_idx].X
        except Exception as e:
            print(f"选择最优解时出错: {str(e)}，返回第一个解")
            # 返回第一个可行解或默认解
            if res.opt and len(res.opt) > 0:
                return res.opt[0].X
            else:
                return np.zeros(time_slots)
    
    def _get_electricity_price(self, hour):
        """获取当前时段电价"""
        for period, (start, end, price) in self.price_periods.items():
            if start <= hour < end:
                return price
        return 0.5  # 默认价格
    
    def evaluate(self, demand: List[float], schedule: List[float]):
        """评估优化效果"""
        if schedule is None:
            print("调度计划为空，无法评估")
            return None
            
        optimized_cost = 0
        storage = 0
        loads = []
        for i in range(len(demand)):
            hour = i % 24
            price = self._get_electricity_price(hour)
            net_power = demand[i] * self.peak_power + schedule[i]
            optimized_cost += net_power * price
            loads.append(net_power)
            storage += schedule[i]
            storage = max(0, min(self.storage_capacity, storage))
        
        # 计算指标
        cost_reduction = (self.original_cost - optimized_cost) / self.original_cost * 100
        peak_loads = [loads[i] for i in range(len(loads)) 
                     if self._is_peak_time(i % 24)]
        optimized_fluctuation = np.std(peak_loads) / np.mean(peak_loads)
        fluctuation_reduction = (self.original_fluctuation - optimized_fluctuation) / self.original_fluctuation * 100
        
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
    
    def visualize(self, demand: List[float], loads: List[float], schedule: List[float]):
        """可视化优化结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 原始需求与优化后负荷
        plt.subplot(3, 1, 1)
        original_loads = [d * self.peak_power for d in demand]
        plt.plot(original_loads, label='原始负荷')
        plt.plot(loads, label='优化后负荷')
        
        # 标记峰时段
        for hour in range(24):
            if self._is_peak_time(hour):
                plt.axvspan(hour, hour+1, alpha=0.1, color='red')
        
        plt.title('负荷优化前后对比')
        plt.xlabel('时间(小时)')
        plt.ylabel('负荷(kW)')
        plt.legend()
        plt.grid()
        
        # 2. 储能充放电计划
        plt.subplot(3, 1, 2)
        plt.bar(range(len(schedule)), schedule)
        plt.title('储能充放电计划')
        plt.xlabel('时间(小时)')
        plt.ylabel('充放电功率(kW)')
        plt.grid()
        
        # 3. 储能状态变化
        plt.subplot(3, 1, 3)
        storage_state = np.cumsum(schedule)
        storage_state = np.clip(storage_state, 0, self.storage_capacity)
        plt.plot(storage_state)
        plt.axhline(self.storage_capacity, color='r', linestyle='--')
        plt.title('储能状态变化')
        plt.xlabel('时间(小时)')
        plt.ylabel('储能电量(kWh)')
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('improved_storage_optimization.png')
        print("\n已生成优化结果图: improved_storage_optimization.png")

if __name__ == "__main__":
    try:
        print("初始化优化器...")
        optimizer = EnergyStorageOptimizer(n_chargers=100, storage_capacity=100)
        
        print("模拟充电需求...")
        demand = optimizer.simulate_demand(days=1)
        print(f"模拟完成，共{len(demand)}小时数据")
        print(f"前几个小时需求数据示例: {demand[:5]}")  # 调试信息
        
        print("\n开始多目标优化...")
        schedule = optimizer.optimize_schedule(demand)
        print("优化完成!")
        
        print("\n评估优化结果...")
        if schedule is not None:
            results = optimizer.evaluate(demand, schedule)
            
            # 检查是否达到目标
            if results is not None and results['cost_reduction'] >= 20 and results['fluctuation_reduction'] >= 50:
                print("\n优化目标已达成!")
            else:
                print("\n优化目标未完全达成，建议进一步调整参数")
        else:
            print("优化未找到可行解，无法评估结果")
            
        # 保持图形显示
        plt.show()
        
问题出在代码中有一段未正确闭合的字符串，导致了语法错误。以下是修复后的完整代码：

```python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as minimize_moo
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
    """改进版储能型充电桩动态调度优化模型"""
    
    def __init__(self, n_chargers=100, storage_capacity=100):  
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
        self.max_charge_power = 80  # 最大充电功率(kW)
        self.max_discharge_power = 100  # 最大放电功率(kW)
        
        # 电价时段划分 (小时)
        self.price_periods = {
            'valley': (0, 8, 0.3),   # 谷电时段: 0-8点, 0.3元/kWh
            'flat': (8, 16, 0.5),     # 平电时段: 8-16点, 0.5元/kWh
            'peak': (16, 24, 0.8)     # 峰电时段: 16-24点, 0.8元/kWh
        }
        
        # 初始化历史数据
        self.original_cost = None
        self.original_fluctuation = None
    
    def simulate_demand(self, days=1):
        """模拟充电需求(默认1天24小时)"""
        demand = []
        for _ in range(days):
            for hour in range(24):
                if self._is_peak_time(hour):
                    # 峰时段需求较高
                    demand.append(random.randint(40, 60) * self.n_chargers / 100)
                else:
                    # 其他时段需求较低
                    demand.append(random.randint(15, 35) * self.n_chargers / 100)
        return np.array(demand)
    
    def _is_peak_time(self, hour):
        """判断是否为峰时段"""
        return self.price_periods['peak'][0] <= hour < self.price_periods['peak'][1]
    
    def optimize_schedule(self, demand: List[float]):
        """多目标优化储能充放电计划"""
        time_slots = len(demand)
        print(f"时间槽数量: {time_slots}")  # 调试信息
        
        # 定义优化问题
        class StorageProblem(Problem):
            def __init__(self, parent):
                super().__init__(
                    n_var=time_slots,
                    n_obj=2,  # 两个目标：最小化成本，最小化波动
                    n_constr=3,  # 三个约束
                    xl=-parent.max_discharge_power,
                    xu=parent.max_charge_power
                )
                self.parent = parent
            
            def _evaluate(self, x, out, *args, **kwargs):
                # 处理批量输入 (x可能是2D数组)
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                
                n_individuals = x.shape[0]
                total_costs = np.zeros(n_individuals)
                fluctuations = np.zeros(n_individuals)
                g1 = np.zeros(n_individuals)
                g2 = np.zeros(n_individuals)
                g3 = np.zeros(n_individuals)
                
                for i in range(n_individuals):
                    # 向量化计算
                    hours = np.arange(time_slots) % 24
                    prices = np.array([self.parent._get_electricity_price(h) for h in hours])
                    
                    # 计算净功率和成本
                    net_powers = demand * self.parent.peak_power + x[i]
                    costs = net_powers * prices
                    total_costs[i] = np.sum(costs)
                    
                    # 计算储能状态
                    storage_states = np.cumsum(x[i])
                    storage_states = np.clip(storage_states, 0, self.parent.storage_capacity)
                    
                    # 计算峰时段负荷波动
                    peak_mask = np.array([self.parent._is_peak_time(h) for h in hours])
                    peak_loads = net_powers[peak_mask]
                    if len(peak_loads) > 0:
                        fluctuations[i] = np.std(peak_loads) / np.mean(peak_loads)
                    
                    # 计算约束
                    g1[i] = abs(storage_states[-1]) - self.parent.storage_capacity
                    g2[i] = np.std(net_powers) / np.mean(net_powers) - 0.3
                    g3[i] = (self.parent.original_cost - total_costs[i]) / self.parent.original_cost - 0.2
                
                out["F"] = np.column_stack([total_costs, fluctuations * 1000])
                out["G"] = np.column_stack([g1, g2, g3])
        
        # 记录原始数据
        self.original_cost = sum(d * self.peak_power * self._get_electricity_price(i % 24) 
                               for i, d in enumerate(demand))
        original_loads = [d * self.peak_power for d in demand]
        peak_loads = [load for i, load in enumerate(original_loads) 
                     if self._is_peak_time(i % 24)]
        self.original_fluctuation = np.std(peak_loads) / np.mean(peak_loads)
        
        # 运行多目标优化
        print("配置优化算法...", flush=True)
        problem = StorageProblem(self)
        algorithm = NSGA2(
            pop_size=100,  # 增加种群大小
            eliminate_duplicates=True
        )
        
        print("开始优化(最多10代)...", flush=True)
        res = minimize_moo(
            problem, 
            algorithm, 
            ('n_gen', 10),  # 增加代数
            verbose=True,
            callback=lambda algorithm: print(f"已完成 {algorithm.n_gen}/10 代", flush=True)
        )
        print("优化完成!", flush=True)
        
        print(f"优化结果类型: {type(res)}")  # 调试信息
        if res is None or res.opt is None or len(res.opt) == 0:
            print("警告: 优化结果中没有可行解")
            # 提供一个默认的解决方案（全零调度）
            return np.zeros(time_slots)
            
        try:
            # 选择最优解(权衡成本和波动)
            best_idx = np.argmin([sol.F[0] * 0.7 + sol.F[1] * 0.3 for sol in res.opt])
            print(f"找到最优解，成本: {res.opt[best_idx].F[0]:.2f}，波动: {res.opt[best_idx].F[1]:.2f}")
            return res.opt[best_idx].X
        except Exception as e:
            print(f"选择最优解时出错: {str(e)}，返回第一个解")
            # 返回第一个可行解或默认解
            if res.opt and len(res.opt) > 0:
                return res.opt[0].X
            else:
                return np.zeros(time_slots)
    
    def _get_electricity_price(self, hour):
        """获取当前时段电价"""
        for period, (start, end, price) in self.price_periods.items():
            if start <= hour < end:
                return price
        return 0.5  # 默认价格
    
    def evaluate(self, demand: List[float], schedule: List[float]):
        """评估优化效果"""
        if schedule is None:
            print("调度计划为空，无法评估")
            return None
            
        optimized_cost = 0
        storage = 0
        loads = []
        for i in range(len(demand)):
            hour = i % 24
            price = self._get_electricity_price(hour)
            net_power = demand[i] * self.peak_power + schedule[i]
            optimized_cost += net_power * price
            loads.append(net_power)
            storage += schedule[i]
            storage = max(0, min(self.storage_capacity, storage))
        
        # 计算指标
        cost_reduction = (self.original_cost - optimized_cost) / self.original_cost * 100
        peak_loads = [loads[i] for i in range(len(loads)) 
                     if self._is_peak_time(i % 24)]
        optimized_fluctuation = np.std(peak_loads) / np.mean(peak_loads)
        fluctuation_reduction = (self.original_fluctuation - optimized_fluctuation) / self.original_fluctuation * 100
        
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
    
    def visualize(self, demand: List[float], loads: List[float], schedule: List[float]):
        """可视化优化结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 原始需求与优化后负荷
        plt.subplot(3, 1, 1)
        original_loads = [d * self.peak_power for d in demand]
        plt.plot(original_loads, label='原始负荷')
        plt.plot(loads, label='优化后负荷')
        
        # 标记峰时段
        for hour in range(24):
            if self._is_peak_time(hour):
                plt.axvspan(hour, hour+1, alpha=0.1, color='red')
        
        plt.title('负荷优化前后对比')
        plt.xlabel('时间(小时)')
        plt.ylabel('负荷(kW)')
        plt.legend()
        plt.grid()
        
        # 2. 储能充放电计划
        plt.subplot(3, 1, 2)
        plt.bar(range(len(schedule)), schedule)
        plt.title('储能充放电计划')
        plt.xlabel('时间(小时)')
        plt.ylabel('充放电功率(kW)')
        plt.grid()
        
        # 3. 储能状态变化
        plt.subplot(3, 1, 3)
        storage_state = np.cumsum(schedule)
        storage_state = np.clip(storage_state, 0, self.storage_capacity)
        plt.plot(storage_state)
        plt.axhline(self.storage_capacity, color='r', linestyle='--')
        plt.title('储能状态变化')
        plt.xlabel('时间(小时)')
        plt.ylabel('储能电量(kWh)')
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('improved_storage_optimization.png')
        print("\n已生成优化结果图: improved_storage_optimization.png")

if __name__ == "__main__":
    try:
        print("初始化优化器...")
        optimizer = EnergyStorageOptimizer(n_chargers=100, storage_capacity=100)
        
        print("模拟充电需求...")
        demand = optimizer.simulate_demand(days=1)
        print(f"模拟完成，共{len(demand)}小时数据")
        print(f"前几个小时需求数据示例: {demand[:5]}")  # 调试信息
        
        print("\n开始多目标优化...")
        schedule = optimizer.optimize_schedule(demand)
        print("优化完成!")
        
        print("\n评估优化结果...")
        if schedule is not None:
            results = optimizer.evaluate(demand, schedule)
            
            # 检查是否达到目标
            if results is not None and results['cost_reduction'] >= 20 and results['fluctuation_reduction'] >= 50:
                print("\n优化目标已达成!")
            else:
                print("\n优化目标未完全达成，建议进一步调整参数")
        else:
            print("优化未找到可行解，无法评估结果")
            
        # 保持图形显示
        plt.show()
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()