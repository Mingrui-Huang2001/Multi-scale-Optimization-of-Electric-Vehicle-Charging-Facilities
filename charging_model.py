import numpy as np
from scipy.stats import poisson

class HouseholdChargingModel:
    def __init__(self, num_chargers=1000, power=7, time_window_prices={'day': 0.8, 'night': 0.3}):
        self.num_chargers = num_chargers
        self.power = power
        self.available_chargers = num_chargers
        self.time_window_prices = time_window_prices
        self.load_history = []
        
    def simulate_shared_usage(self, day_hours=12, lambda_day=15):
        """
        模拟日间共享充电的排队过程
        :param day_hours: 日间时段小时数
        :param lambda_day: 日间充电请求到达率（次/小时）
        :return: 平均等待时间，电网负荷标准差
        """
        # 动态调整到达率（考虑电价影响因子）
        price_factor = 1.5 if 8 <= current_time < 20 else 0.7
        adjusted_lambda = lambda_day * price_factor
        inter_arrival = np.random.exponential(1/adjusted_lambda, 10000)
        arrival_times = np.cumsum(inter_arrival)
        
        # 生成充电时长（正态分布）
        charge_duration = np.abs(np.random.normal(2, 0.5, len(arrival_times)))
        
        # 初始化状态变量
        available_chargers = self.num_chargers
        load_profile = np.zeros(24*60)  # 每分钟负荷
        waiting_times = []
        
        # 离散事件仿真
        current_time = 0
        for i in range(len(arrival_times)):
            # 处理充电结束事件
            while current_time >= arrival_times[i]:
                if available_chargers > 0:
                    available_chargers -= 1
                    end_time = arrival_times[i] + charge_duration[i]
                    # 更新负荷曲线
                    start_idx = int(arrival_times[i]*60)
                    end_idx = int(end_time*60)
                    load_profile[start_idx:end_idx] += self.power
                    waiting_times.append(0)
                else:
                    # 计算等待时间
                    waiting_time = max(0, available_chargers - arrival_times[i])
                    waiting_times.append(waiting_time)
                current_time += 0.1  # 时间步进
        
        avg_wait = np.mean(waiting_times)
        load_std = np.std(load_profile)
        # 实时记录负荷波动
        self.load_history.append({
            'timestep': current_time,
            'load': load_profile[-1],
            'std': load_std
        })
        return avg_wait, load_std

    def calculate_utilization(self, shared_hours=12):
        """计算充电桩利用率提升"""
        base_util = 2/24
        shared_util = (2 + shared_hours*0.7)/24
        return shared_util - base_util

if __name__ == "__main__":
    model = HouseholdChargingModel()
    avg_wait, load_std = model.simulate_shared_usage()
    print(f"平均等待时间：{avg_wait:.2f}小时")
    print(f"负荷标准差：{load_std:.2f}kW")