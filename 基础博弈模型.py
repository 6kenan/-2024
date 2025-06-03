import math
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

# 定义车辆类，方便存储和传递车辆信息
class Vehicle:
    def __init__(self, id, d, v, a_max):
        self.id = id # 原始ID
        self.d = d   # 到达冲突区域的距离 (m)
        self.v = v   # 当前速度 (m/s)
        self.a_max = a_max # 最大加速度 (m/s^2)
        self.t_min = self.calculate_t_min() # 以最大加速度到达的最短时间 (s)
        self.eta = d / v if v > 0 else float('inf') # 预计到达时间 (s)
        
        # GLOSO 和 LOGA 阶段会填充这些值
        self.gloso_id = -1 # 在GLOSO排序后的ID (1, 2, or 3)
        self.t_seq_gloso = 0.0 # GLOSO建议的到达时间
        self.k_loga = 0      # LOGA阶段选择的调整因子
        self.t_tar_loga = 0.0  # LOGA阶段的目标到达时间

    def calculate_t_min(self):
        # 解方程 d = v*t + 0.5*a_max*t^2 for t
        # 0.5*a_max*t^2 + v*t - d = 0
        a = 0.5 * self.a_max
        b = self.v
        c = -self.d
        
        if abs(a) < 1e-9: # 修正：处理a_max可能为0的情况
            if abs(b) < 1e-9: # 如果速度和加速度都为0
                return float('inf') if c < 0 else (0 if c==0 else float('inf')) # 如果d>0则无法到达, d=0则瞬时到达
            if b > 0:
                return -c / b if -c / b >=0 else float('inf')
            else: # b < 0, 除非d=0, 否则无法到达
                return 0 if c == 0 else float('inf')


        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return float('inf') # 无实数解，意味着无法到达
        
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2*a)
        t2 = (-b - sqrt_discriminant) / (2*a)
        
        # 返回非负的最小时间
        valid_times = []
        if t1 >= -1e-9: #允许微小的负值，当作0
            valid_times.append(max(0, t1))
        if t2 >= -1e-9:
            valid_times.append(max(0, t2))
        
        return min(valid_times) if valid_times else float('inf')


    def __repr__(self):
        return (f"Veh(orig_id={self.id}, d={self.d}, v={self.v}, a_max={self.a_max}, "
                f"t_min={self.t_min:.2f}s, eta={self.eta:.2f}s, "
                f"gloso_id={self.gloso_id}, t_seq_gloso={self.t_seq_gloso:.2f}s)")

# --- 模型参数 ---
T_SAFE = 1.5  # 安全时间间隔 (s)
DELTA_T = 0.5 # 时间调整步长 (s)

# 支付函数权重 (alpha_1, alpha_2, alpha_3, alpha_4)
# alpha_1: GLOSO偏差惩罚, alpha_2: 效率收益, alpha_3: 舒适性收益, alpha_4: 安全收益
ALPHAS = {
    "alpha1": 0.5, # r_o
    "alpha2": 0.3, # r_e
    "alpha3": 0.2, # r_c
    "alpha4": 0.1  # r_s (当前简化为0，如果满足基本安全)
}
# DELTA_T_COMFORT = 0.2 # 如果使用更复杂的r_s，可以取消注释

# --- GLOSO 阶段 ---
def run_gloso(initial_vehicles):
    """
    执行GLOSO阶段，确定排序和计划到达时间。
    """
    # 1. 按ETA排序
    sorted_vehicles = sorted(initial_vehicles, key=lambda veh: veh.eta)
    
    for i, veh in enumerate(sorted_vehicles):
        veh.gloso_id = i + 1

    t_seq_gloso_list = []
    for i, veh in enumerate(sorted_vehicles):
        if i == 0: 
            veh.t_seq_gloso = max(veh.eta, veh.t_min)
        else:
            prev_veh_t_seq = sorted_vehicles[i-1].t_seq_gloso
            veh.t_seq_gloso = max(prev_veh_t_seq + T_SAFE, veh.eta, veh.t_min)
        t_seq_gloso_list.append(veh.t_seq_gloso)
        
    return sorted_vehicles, t_seq_gloso_list

# --- LOGA 阶段 ---
def calculate_t_tar_profile(gloso_vehicles, k_profile):
    t_tar_profile = []
    for i in range(len(gloso_vehicles)):
        veh = gloso_vehicles[i] 
        k_val = k_profile[i]
        t_tar = veh.t_seq_gloso + k_val * DELTA_T
        t_tar_profile.append(t_tar)
    return tuple(t_tar_profile)

def is_profile_legal(t_tar_profile, gloso_vehicles):
    for i in range(len(gloso_vehicles)):
        if t_tar_profile[i] < gloso_vehicles[i].t_min - 1e-6: 
            return False
            
    if len(gloso_vehicles) > 1:
        if t_tar_profile[1] - t_tar_profile[0] < T_SAFE - 1e-6:
            return False
    if len(gloso_vehicles) > 2:
        if t_tar_profile[2] - t_tar_profile[1] < T_SAFE - 1e-6:
            return False
    return True

def calculate_payoff_for_vehicle(veh_idx_in_gloso, 
                                 t_tar_profile,    
                                 k_profile,        
                                 gloso_vehicles):  
    veh = gloso_vehicles[veh_idx_in_gloso]
    t_tar_self = t_tar_profile[veh_idx_in_gloso]

    r_o = -(t_tar_self - veh.t_seq_gloso)**2
    r_e = -t_tar_self
    r_c = -(t_tar_self - veh.eta)**2 
    r_s = 0.0 

    payoff = (ALPHAS["alpha1"] * r_o +
              ALPHAS["alpha2"] * r_e +
              ALPHAS["alpha3"] * r_c +
              ALPHAS["alpha4"] * r_s)
    return payoff

def find_nash_equilibria(legal_payoffs, gloso_vehicles):
    nash_equilibria = []
    possible_k_values = [-1, 0, 1]

    for k_profile_star, payoffs_star in legal_payoffs.items():
        is_ne = True
        for player_idx in range(len(gloso_vehicles)): 
            current_payoff_for_player = payoffs_star[player_idx]
            
            for k_alternative in possible_k_values:
                if k_alternative == k_profile_star[player_idx]:
                    continue 

                k_profile_deviated_list = list(k_profile_star)
                k_profile_deviated_list[player_idx] = k_alternative
                k_profile_deviated = tuple(k_profile_deviated_list)

                if k_profile_deviated in legal_payoffs:
                    payoffs_deviated = legal_payoffs[k_profile_deviated]
                    payoff_after_deviation_for_player = payoffs_deviated[player_idx]
                    if payoff_after_deviation_for_player > current_payoff_for_player + 1e-9: # 允许微小误差比较
                        is_ne = False
                        break 
            if not is_ne:
                break 
        
        if is_ne:
            nash_equilibria.append(k_profile_star)
            
    return nash_equilibria

# --- 可视化函数 ---
def plot_nash_equilibria_payoffs(nash_equilibria_profiles, legal_payoffs_loga, gloso_vehicles):
    if not nash_equilibria_profiles:
        print("没有纳什均衡可供可视化。")
        return

    # 设置matplotlib支持中文的字体
    # 这需要系统中安装了支持中文的字体，例如'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
    # 如果以下字体不可用，matplotlib会回退到默认字体，中文可能显示为方框
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    num_ne = len(nash_equilibria_profiles)
    num_vehicles = len(gloso_vehicles)

    payoff_values = np.zeros((num_vehicles, num_ne))
    ne_labels = []

    for i, ne_k_profile in enumerate(nash_equilibria_profiles):
        ne_labels.append(str(ne_k_profile))
        payoffs = legal_payoffs_loga[ne_k_profile]
        for j in range(num_vehicles):
            payoff_values[j, i] = payoffs[j]

    x = np.arange(num_ne)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(max(10, num_ne * 1.5), 6)) # 动态调整图形宽度

    rects_list = []
    for i in range(num_vehicles):
        # 计算每个条形组中当前条形的位置
        offset = width * (i - (num_vehicles - 1) / 2)
        rects = ax.bar(x + offset, payoff_values[i], width, label=f'车辆 (GLOSO ID {i+1}) - 原ID {gloso_vehicles[i].id}')
        rects_list.append(rects)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('支付值 (Payoff)')
    ax.set_xlabel('纳什均衡策略组合 ($k$-Profile)')
    ax.set_title('纳什均衡下各车辆的支付值')
    ax.set_xticks(x)
    ax.set_xticklabels(ne_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    
    #尝试显示图形
    try:
        plt.show()
        print("\n图形已生成并尝试显示。如果未显示，请检查您的环境是否支持matplotlib图形界面。")
    except Exception as e:
        print(f"\n生成图形时出错或无法显示: {e}")
        print("请确保您的环境已安装matplotlib并支持GUI。")


# --- 主执行流程 ---
if __name__ == "__main__":
    initial_vehicles_data = [
        Vehicle(id=1, d=100, v=10, a_max=2),
        Vehicle(id=2, d=80, v=12, a_max=2.5),
        Vehicle(id=3, d=120, v=11, a_max=1.8)
    ]
    print("--- 初始车辆状态 (计算得到的 t_min 和 ETA) ---")
    for veh in initial_vehicles_data:
        print(f"Veh ID {veh.id}: d={veh.d}, v={veh.v}, a_max={veh.a_max}, "
              f"t_min={veh.t_min:.2f}s, ETA={veh.eta:.2f}s")
    print("-" * 50)

    gloso_ordered_vehicles, t_seq_gloso_values = run_gloso(initial_vehicles_data)
    
    print("--- GLOSO 阶段结果 ---")
    print("GLOSO 通行顺序 (按原始ID):", [veh.id for veh in gloso_ordered_vehicles])
    print("GLOSO 计划到达时间序列 T_seq*:", [f"{t:.2f}s" for t in t_seq_gloso_values])
    print("详细车辆GLOSO信息:")
    for veh in gloso_ordered_vehicles:
        print(veh)
    print("-" * 50)

    k_values = [-1, 0, 1]
    all_k_profiles = list(product(k_values, repeat=len(gloso_ordered_vehicles)))
    legal_payoffs_loga = {} 
    
    print("--- LOGA 阶段: 合法策略及其支付 ---")
    print(f"{'k-Profile':<12} | {'t_tar Profile (s)':<25} | {'Legality':<8} | {'Payoffs (R1, R2, R3)':<30}")
    print("-" * 80)

    for k_profile in all_k_profiles:
        t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, k_profile)
        is_legal = is_profile_legal(t_tar_profile, gloso_ordered_vehicles)
        
        payoffs_for_profile_str = "N/A (Illegal)"
        if is_legal:
            payoffs = []
            for i in range(len(gloso_ordered_vehicles)):
                payoff_veh_i = calculate_payoff_for_vehicle(
                    veh_idx_in_gloso=i,
                    t_tar_profile=t_tar_profile,
                    k_profile=k_profile,
                    gloso_vehicles=gloso_ordered_vehicles
                )
                payoffs.append(payoff_veh_i)
            legal_payoffs_loga[k_profile] = tuple(payoffs)
            payoffs_for_profile_str = f"({payoffs[0]:.3f}, {payoffs[1]:.3f}, {payoffs[2]:.3f})"

        t_tar_profile_str = f"({t_tar_profile[0]:.2f}, {t_tar_profile[1]:.2f}, {t_tar_profile[2]:.2f})"
        print(f"{str(k_profile):<12} | {t_tar_profile_str:<25} | {str(is_legal):<8} | {payoffs_for_profile_str:<30}")

    print("-" * 80)
    print(f"总共 {len(all_k_profiles)} 种策略组合, 其中 {len(legal_payoffs_loga)} 种是合法的。")
    print("-" * 50)
    
    nash_equilibria_profiles = find_nash_equilibria(legal_payoffs_loga, gloso_ordered_vehicles)
    
    print("--- 纳什均衡结果 ---")
    if not nash_equilibria_profiles:
        print("没有找到纯策略纳什均衡。")
    else:
        print(f"找到了 {len(nash_equilibria_profiles)} 个纯策略纳什均衡:")
        print(f"{'NE k-Profile':<15} | {'t_tar Profile (s)':<25} | {'Payoffs (R1, R2, R3)'}")
        print("-" * 70)
        for ne_k_profile in nash_equilibria_profiles:
            ne_t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, ne_k_profile)
            ne_payoffs = legal_payoffs_loga[ne_k_profile]
            
            ne_t_tar_str = f"({ne_t_tar_profile[0]:.2f}, {ne_t_tar_profile[1]:.2f}, {ne_t_tar_profile[2]:.2f})"
            ne_payoffs_str = f"({ne_payoffs[0]:.3f}, {ne_payoffs[1]:.3f}, {ne_payoffs[2]:.3f})"
            print(f"{str(ne_k_profile):<15} | {ne_t_tar_str:<25} | {ne_payoffs_str}")
    print("-" * 50)

    # 5. 可视化纳什均衡支付
    if nash_equilibria_profiles:
        plot_nash_equilibria_payoffs(nash_equilibria_profiles, legal_payoffs_loga, gloso_ordered_vehicles)