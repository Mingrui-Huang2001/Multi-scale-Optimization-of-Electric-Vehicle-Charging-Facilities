# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import matplotlib
import sys
from charging_model_1 import ChargingModel
from fast_charger_optimization import DemandPredictor, GridSafetyAnalyzer

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

class CoordinatedCharging:
    """充电桩共享与电网协同优化系统"""
    
    def __init__(self, home_chargers=1000, fast_chargers=500):
        # 初始化家庭充电模型
        self.home_model = ChargingModel(n_chargers=home_chargers, power=7)
        self.home_model.set_user_behavior(lambda_night=0.3, lambda_day=0.2)
        
        # 初始化快充桩模型
        self.fast_predictor = DemandPredictor(n_clusters=5)
        self.grid_analyzer = GridSafetyAnalyzer(base_load=5000)
        
        # 协同参数
        self.transfer_rate = 0.2  # 家庭充电可转为快充的比例
        self.price_ratio = 1.5    # 快充/家庭充电价格比
        self.max_grid_load = 10000  # 电网最大允许负荷(kW)
        
        # 优化历史记录
        self.optimization_history = []
    
    def predict_demand(self, hour):
        """预测小时级充电需求"""
        # 家庭充电需求
        home_demand = self.home_model.predict_demand(hour)
        
        # 快充需求(基于历史数据)
        fast_demand = home_demand * 0.3  # 假设快充需求是家庭的30%
        
        return {
            'home': home_demand,
            'fast': fast_demand,
            'total': home_demand + fast_demand
        }
    
    def optimize_scheduling(self, demand_prediction):
        """优化充电调度方案(确保电网不过载)"""
        # 提取预测需求
        home_demand = demand_prediction['home']
        fast_demand = demand_prediction['fast']
        
        # 定义安全优化问题
        def objective(x):
            # x[0]: 转移量, x[1]: 快充价格系数, x[2]: 功率调整因子
            transferred = x[0]
            price_factor = x[1]
            power_adjust = x[2]
            
            # 调整后的功率
            adjusted_home_power = self.home_model.power * power_adjust
            adjusted_fast_power = 60 * power_adjust  # 基础快充功率60kW
            
            # 计算电网负荷
            home_load = (home_demand - transferred) * adjusted_home_power
            fast_load = (fast_demand + transferred) * adjusted_fast_power
            total_load = home_load + fast_load
            
            # 严格约束惩罚(平方惩罚使过载不可行)
            overload_penalty = max(0, total_load - self.max_grid_load) ** 2 * 1e6
            
            # 收益计算
            home_revenue = (home_demand - transferred) * 1.0
            fast_revenue = (fast_demand + transferred) * 1.0 * price_factor
            
            # 目标: 最大化收益同时严格避免过载
            return -(home_revenue + fast_revenue) + overload_penalty
            
        def grid_safety_constraint(x):
            """电网安全硬约束"""
            transferred = x[0]
            power_adjust = x[2]
            home_load = (home_demand - transferred) * self.home_model.power * power_adjust
            fast_load = (fast_demand + transferred) * 60 * power_adjust
            return self.max_grid_load - (home_load + fast_load)  # 必须≥0
        
        # 约束条件
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # 转移量≥0
            {'type': 'ineq', 'fun': lambda x: home_demand - x[0]},  # 转移量≤家庭需求
            {'type': 'ineq', 'fun': lambda x: x[1] - 1.0},  # 价格系数≥1
            {'type': 'ineq', 'fun': lambda x: 2.0 - x[1]},   # 价格系数≤2
            {'type': 'ineq', 'fun': lambda x: x[2]},  # 功率调整≥0
            {'type': 'ineq', 'fun': lambda x: 1.5 - x[2]}  # 功率调整≤1.5
        ]
        
        # 初始猜测 [转移量, 价格系数, 功率调整]
        x0 = [
            min(home_demand * 0.2, fast_demand * 0.3),  # 转移量
            1.5,  # 价格系数
            1.0  # 功率调整(1.0=100%功率)
        ]
        
        # 运行优化
        res = minimize(objective, x0, constraints=constraints, 
                      method='SLSQP', options={'maxiter': 100})
        
        return {
            'transfer': res.x[0],
            'price_factor': res.x[1],
            'home_load': (home_demand - res.x[0]) * self.home_model.power,
            'fast_load': (fast_demand + res.x[0]) * 60,
            'success': res.success
        }
    
    def simulate_day(self):
        """模拟24小时运行"""
        hourly_results = []
        
        for hour in range(24):
            # 预测需求
            demand = self.predict_demand(hour)
            
            # 优化调度
            schedule = self.optimize_scheduling(demand)
            
            # 记录结果
            hourly_results.append({
                'hour': hour,
                'demand': demand,
                'schedule': schedule,
                'grid_load': schedule['home_load'] + schedule['fast_load']
            })
        
        return hourly_results
    
    def evaluate_performance(self, hourly_results):
        """评估系统性能"""
        # 提取关键指标
        loads = [r['grid_load'] for r in hourly_results]
        transfers = [r['schedule']['transfer'] for r in hourly_results]
        revenues = [
            (r['demand']['home'] - r['schedule']['transfer']) * 1.0 +
            (r['demand']['fast'] + r['schedule']['transfer']) * 1.0 * r['schedule']['price_factor']
            for r in hourly_results
        ]
        
        # 计算统计量
        load_stats = {
            'max': max(loads),
            'min': min(loads),
            'avg': np.mean(loads),
            'std': np.std(loads),
            'peak_valley': max(loads) - min(loads)
        }
        
        transfer_stats = {
            'total': sum(transfers),
            'max': max(transfers),
            'avg': np.mean(transfers)
        }
        
        revenue_stats = {
            'total': sum(revenues),
            'max': max(revenues),
            'avg': np.mean(revenues)
        }
        
        return {
            'load': load_stats,
            'transfer': transfer_stats,
            'revenue': revenue_stats,
            'grid_safety': {
                'overload': any(l > self.max_grid_load for l in loads),
                'max_load_percent': max(loads) / self.max_grid_load
            }
        }
    
    def visualize_results(self, hourly_results, save_path='coordinated_results.png'):
        """可视化协调优化结果"""
        plt.figure(figsize=(15, 8))
        
        # 1. 负荷曲线
        plt.subplot(2, 2, 1)
        hours = range(24)
        loads = [r['grid_load'] for r in hourly_results]
        home_loads = [r['schedule']['home_load'] for r in hourly_results]
        fast_loads = [r['schedule']['fast_load'] for r in hourly_results]
        
        plt.plot(hours, loads, 'k-', label='总负荷')
        plt.plot(hours, home_loads, 'b--', label='家庭充电负荷')
        plt.plot(hours, fast_loads, 'r--', label='快充负荷')
        plt.axhline(self.max_grid_load, color='g', linestyle=':', label='电网容量')
        plt.xlabel('时间(小时)')
        plt.ylabel('负荷(kW)')
        plt.title('24小时电网负荷曲线')
        plt.legend()
        plt.grid()
        
        # 2. 需求转移量
        plt.subplot(2, 2, 2)
        transfers = [r['schedule']['transfer'] for r in hourly_results]
        plt.bar(hours, transfers)
        plt.xlabel('时间(小时)')
        plt.ylabel('转移量(辆)')
        plt.title('家庭充电向快充转移量')
        plt.grid()
        
        # 3. 价格系数
        plt.subplot(2, 2, 3)
        prices = [r['schedule']['price_factor'] for r in hourly_results]
        plt.plot(hours, prices, 'm-')
        plt.xlabel('时间(小时)')
        plt.ylabel('价格系数')
        plt.title('快充动态价格系数')
        plt.grid()
        
        # 4. 收益曲线
        plt.subplot(2, 2, 4)
        revenues = [
            (r['demand']['home'] - r['schedule']['transfer']) * 1.0 +
            (r['demand']['fast'] + r['schedule']['transfer']) * 1.0 * r['schedule']['price_factor']
            for r in hourly_results
        ]
        plt.plot(hours, revenues, 'g-')
        plt.xlabel('时间(小时)')
        plt.ylabel('收益(元)')
        plt.title('每小时收益曲线')
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"已保存可视化结果: {save_path}")

if __name__ == "__main__":
    # 初始化协同系统
    system = CoordinatedCharging(home_chargers=1000, fast_chargers=500)
    
    # 运行24小时模拟
    print("正在运行协同优化模拟...")
    results = system.simulate_day()
    
    # 评估性能
    metrics = system.evaluate_performance(results)
    print("\n性能评估结果:")
    print(f"总收益: {metrics['revenue']['total']:.2f} 元")
    print(f"最大负荷: {metrics['load']['max']:.2f} kW ({metrics['grid_safety']['max_load_percent']:.1%} of capacity)")
    print(f"总转移量: {metrics['transfer']['total']:.0f} 辆次")
    print(f"电网过载: {'是' if metrics['grid_safety']['overload'] else '否'}")
    
    # 可视化结果
    system.visualize_results(results)
    
    # 保存详细结果
    import pandas as pd
    df = pd.DataFrame([{
        'hour': r['hour'],
        'home_demand': r['demand']['home'],
        'fast_demand': r['demand']['fast'],
        'transfer': r['schedule']['transfer'],
        'price_factor': r['schedule']['price_factor'],
        'home_load': r['schedule']['home_load'],
        'fast_load': r['schedule']['fast_load'],
        'total_load': r['grid_load']
    } for r in results])
    df.to_csv('hourly_charging_data.csv', index=False)
    print("已保存详细数据: hourly_charging_data.csv")