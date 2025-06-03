import math
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

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

# --- GLOSO 阶段 ---
def run_gloso(initial_vehicles, t_safe):
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
            veh.t_seq_gloso = max(prev_veh_t_seq + t_safe, veh.eta, veh.t_min)
        t_seq_gloso_list.append(veh.t_seq_gloso)
        
    return sorted_vehicles, t_seq_gloso_list

# --- LOGA 阶段 ---
def calculate_t_tar_profile(gloso_vehicles, k_profile, delta_t):
    t_tar_profile = []
    for i in range(len(gloso_vehicles)):
        veh = gloso_vehicles[i] 
        k_val = k_profile[i]
        t_tar = veh.t_seq_gloso + k_val * delta_t
        t_tar_profile.append(t_tar)
    return tuple(t_tar_profile)

def is_profile_legal(t_tar_profile, gloso_vehicles, t_safe):
    for i in range(len(gloso_vehicles)):
        if t_tar_profile[i] < gloso_vehicles[i].t_min - 1e-6: 
            return False
            
    if len(gloso_vehicles) > 1:
        if t_tar_profile[1] - t_tar_profile[0] < t_safe - 1e-6:
            return False
    if len(gloso_vehicles) > 2:
        if t_tar_profile[2] - t_tar_profile[1] < t_safe - 1e-6:
            return False
    return True

def calculate_payoff_for_vehicle(veh_idx_in_gloso, 
                                 t_tar_profile,    
                                 k_profile,        
                                 gloso_vehicles,
                                 alphas):  
    veh = gloso_vehicles[veh_idx_in_gloso]
    t_tar_self = t_tar_profile[veh_idx_in_gloso]

    r_o = -(t_tar_self - veh.t_seq_gloso)**2
    r_e = -t_tar_self
    r_c = -(t_tar_self - veh.eta)**2 
    r_s = 0.0 

    payoff = (alphas["alpha1"] * r_o +
              alphas["alpha2"] * r_e +
              alphas["alpha3"] * r_c +
              alphas["alpha4"] * r_s)
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

# --- 参数敏感性分析 ---
def analyze_nash_equilibria_for_parameters(vehicles, t_safe_values, delta_t_values, alphas):
    results = {}
    
    for t_safe in t_safe_values:
        results[t_safe] = {}
        for delta_t in delta_t_values:
            # 复制车辆以避免修改原始数据
            vehicles_copy = [Vehicle(v.id, v.d, v.v, v.a_max) for v in vehicles]
            
            # 运行GLOSO阶段
            gloso_ordered_vehicles, _ = run_gloso(vehicles_copy, t_safe)
            
            # 生成所有可能的k值组合
            k_values = [-1, 0, 1]
            all_k_profiles = list(product(k_values, repeat=len(gloso_ordered_vehicles)))
            
            # 计算合法策略及其支付
            legal_payoffs_loga = {}
            for k_profile in all_k_profiles:
                t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, k_profile, delta_t)
                is_legal = is_profile_legal(t_tar_profile, gloso_ordered_vehicles, t_safe)
                
                if is_legal:
                    payoffs = []
                    for i in range(len(gloso_ordered_vehicles)):
                        payoff_veh_i = calculate_payoff_for_vehicle(
                            veh_idx_in_gloso=i,
                            t_tar_profile=t_tar_profile,
                            k_profile=k_profile,
                            gloso_vehicles=gloso_ordered_vehicles,
                            alphas=alphas
                        )
                        payoffs.append(payoff_veh_i)
                    legal_payoffs_loga[k_profile] = tuple(payoffs)
            
            # 寻找纳什均衡
            nash_equilibria_profiles = find_nash_equilibria(legal_payoffs_loga, gloso_ordered_vehicles)
            
            # 存储结果
            results[t_safe][delta_t] = {
                'nash_equilibria': nash_equilibria_profiles,
                'legal_payoffs': legal_payoffs_loga,
                'gloso_vehicles': gloso_ordered_vehicles,
                'num_legal_strategies': len(legal_payoffs_loga),
                'num_nash_equilibria': len(nash_equilibria_profiles)
            }
    
    return results

# --- 可视化函数 ---
def plot_comprehensive_stability_analysis(results, t_safe_values, delta_t_values):
    # 创建热力图数据
    nash_counts = np.zeros((len(t_safe_values), len(delta_t_values)))
    legal_counts = np.zeros((len(t_safe_values), len(delta_t_values)))
    stability_ratio = np.zeros((len(t_safe_values), len(delta_t_values)))
    
    for i, t_safe in enumerate(t_safe_values):
        for j, delta_t in enumerate(delta_t_values):
            nash_counts[i, j] = results[t_safe][delta_t]['num_nash_equilibria']
            legal_counts[i, j] = results[t_safe][delta_t]['num_legal_strategies']
            if legal_counts[i, j] > 0:
                stability_ratio[i, j] = nash_counts[i, j] / legal_counts[i, j]
    
    # 创建一个3x1的子图布局
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 1. 纳什均衡数量热力图
    sns.heatmap(nash_counts, annot=True, fmt=".0f", cmap="YlGnBu", 
                xticklabels=[f"{dt:.2f}" for dt in delta_t_values],
                yticklabels=[f"{ts:.2f}" for ts in t_safe_values],
                ax=axes[0])
    axes[0].set_title("纳什均衡数量")
    axes[0].set_xlabel("时间调整步长 (DELTA_T)")
    axes[0].set_ylabel("安全时间间隔 (T_SAFE)")
    
    # 2. 合法策略数量热力图
    sns.heatmap(legal_counts, annot=True, fmt=".0f", cmap="YlOrRd", 
                xticklabels=[f"{dt:.2f}" for dt in delta_t_values],
                yticklabels=[f"{ts:.2f}" for ts in t_safe_values],
                ax=axes[1])
    axes[1].set_title("合法策略数量")
    axes[1].set_xlabel("时间调整步长 (DELTA_T)")
    axes[1].set_ylabel("安全时间间隔 (T_SAFE)")
    
    # 3. 稳定性比率热力图 (纳什均衡数量/合法策略数量)
    sns.heatmap(stability_ratio, annot=True, fmt=".2f", cmap="RdYlGn", 
                xticklabels=[f"{dt:.2f}" for dt in delta_t_values],
                yticklabels=[f"{ts:.2f}" for ts in t_safe_values],
                ax=axes[2])
    axes[2].set_title("稳定性比率 (纳什均衡数量/合法策略数量)")
    axes[2].set_xlabel("时间调整步长 (DELTA_T)")
    axes[2].set_ylabel("安全时间间隔 (T_SAFE)")
    
    plt.tight_layout()
    plt.savefig("comprehensive_stability_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap_analysis(results, t_safe_values, delta_t_values):
    # 创建一个自定义的颜色映射，从红色（0）到绿色（1）
    cmap = LinearSegmentedColormap.from_list('RdYlGn', ['red', 'yellow', 'green'])
    
    # 提取最优纳什均衡的数据
    best_ne_data = np.zeros((len(t_safe_values), len(delta_t_values)))
    best_ne_k_profiles = np.empty((len(t_safe_values), len(delta_t_values)), dtype=object)
    
    for i, t_safe in enumerate(t_safe_values):
        for j, delta_t in enumerate(delta_t_values):
            nash_equilibria = results[t_safe][delta_t]['nash_equilibria']
            legal_payoffs = results[t_safe][delta_t]['legal_payoffs']
            
            if nash_equilibria:
                # 计算每个纳什均衡的总支付
                total_payoffs = [sum(legal_payoffs[ne]) for ne in nash_equilibria]
                best_idx = np.argmax(total_payoffs)
                best_ne = nash_equilibria[best_idx]
                best_ne_data[i, j] = total_payoffs[best_idx]
                best_ne_k_profiles[i, j] = best_ne
            else:
                best_ne_data[i, j] = float('nan')  # 没有纳什均衡
                best_ne_k_profiles[i, j] = None
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制热力图
    im = ax.imshow(best_ne_data, cmap=cmap, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('最优纳什均衡的总支付')
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(delta_t_values)))
    ax.set_yticks(np.arange(len(t_safe_values)))
    ax.set_xticklabels([f"{dt:.2f}" for dt in delta_t_values])
    ax.set_yticklabels([f"{ts:.2f}" for ts in t_safe_values])
    
    # 添加标签和标题
    ax.set_xlabel('时间调整步长 (DELTA_T)')
    ax.set_ylabel('安全时间间隔 (T_SAFE)')
    ax.set_title('参数敏感性分析: 最优纳什均衡的总支付')
    
    # 在每个单元格中添加最优纳什均衡的k-profile
    for i in range(len(t_safe_values)):
        for j in range(len(delta_t_values)):
            if best_ne_k_profiles[i, j] is not None:
                text = ax.text(j, i, str(best_ne_k_profiles[i, j]),
                               ha="center", va="center", color="black", fontsize=8)
            else:
                text = ax.text(j, i, "无NE",
                               ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig("best_nash_equilibria_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_recommendations(results, t_safe_values, delta_t_values, vehicles):
    # 找出具有最多纳什均衡的参数组合
    max_ne_count = 0
    max_ne_params = None
    
    # 找出具有最高总支付的参数组合
    max_total_payoff = float('-inf')
    max_payoff_params = None
    max_payoff_ne = None
    
    for t_safe in t_safe_values:
        for delta_t in delta_t_values:
            result = results[t_safe][delta_t]
            nash_equilibria = result['nash_equilibria']
            legal_payoffs = result['legal_payoffs']
            
            # 更新最多纳什均衡的参数
            if len(nash_equilibria) > max_ne_count:
                max_ne_count = len(nash_equilibria)
                max_ne_params = (t_safe, delta_t)
            
            # 计算每个纳什均衡的总支付并找出最高的
            for ne in nash_equilibria:
                total_payoff = sum(legal_payoffs[ne])
                if total_payoff > max_total_payoff:
                    max_total_payoff = total_payoff
                    max_payoff_params = (t_safe, delta_t)
                    max_payoff_ne = ne
    
    # 生成分析报告
    report = "\n=== 参数敏感性分析报告 ===\n\n"
    
    # 1. 参数范围概述
    report += "1. 参数范围概述:\n"
    report += f"   - 安全时间间隔 (T_SAFE): {min(t_safe_values):.2f}s 到 {max(t_safe_values):.2f}s\n"
    report += f"   - 时间调整步长 (DELTA_T): {min(delta_t_values):.2f}s 到 {max(delta_t_values):.2f}s\n\n"
    
    # 2. 纳什均衡分布
    report += "2. 纳什均衡分布:\n"
    ne_counts = {}
    for t_safe in t_safe_values:
        for delta_t in delta_t_values:
            count = results[t_safe][delta_t]['num_nash_equilibria']
            ne_counts[(t_safe, delta_t)] = count
    
    # 按纳什均衡数量排序
    sorted_counts = sorted(ne_counts.items(), key=lambda x: x[1], reverse=True)
    for (t_safe, delta_t), count in sorted_counts[:5]:  # 只显示前5个
        report += f"   - T_SAFE={t_safe:.2f}s, DELTA_T={delta_t:.2f}s: {count}个纳什均衡\n"
    report += "\n"
    
    # 3. 最佳参数推荐
    report += "3. 最佳参数推荐:\n"
    if max_ne_params:
        report += f"   - 最多纳什均衡的参数组合: T_SAFE={max_ne_params[0]:.2f}s, DELTA_T={max_ne_params[1]:.2f}s (共{max_ne_count}个均衡)\n"
    if max_payoff_params:
        report += f"   - 最高总支付的参数组合: T_SAFE={max_payoff_params[0]:.2f}s, DELTA_T={max_payoff_params[1]:.2f}s (总支付={max_total_payoff:.4f})\n"
        report += f"   - 对应的最优纳什均衡k-profile: {max_payoff_ne}\n"
    report += "\n"
    
    # 4. 稳定性分析
    report += "4. 稳定性分析:\n"
    stability_data = {}
    for t_safe in t_safe_values:
        for delta_t in delta_t_values:
            result = results[t_safe][delta_t]
            if result['num_legal_strategies'] > 0:
                stability = result['num_nash_equilibria'] / result['num_legal_strategies']
                stability_data[(t_safe, delta_t)] = stability
    
    # 按稳定性比率排序
    sorted_stability = sorted(stability_data.items(), key=lambda x: x[1], reverse=True)
    for (t_safe, delta_t), ratio in sorted_stability[:5]:  # 只显示前5个
        report += f"   - T_SAFE={t_safe:.2f}s, DELTA_T={delta_t:.2f}s: 稳定性比率={ratio:.4f}\n"
    report += "\n"
    
    # 5. 建议和结论
    report += "5. 建议和结论:\n"
    
    # 根据分析结果给出建议
    if max_payoff_params and max_ne_params:
        if max_payoff_params == max_ne_params:
            report += f"   - 推荐参数组合: T_SAFE={max_payoff_params[0]:.2f}s, DELTA_T={max_payoff_params[1]:.2f}s\n"
            report += "   - 该参数组合既有最多的纳什均衡，又能实现最高的总支付，是最佳选择。\n"
        else:
            report += f"   - 如果优先考虑系统稳定性，推荐: T_SAFE={max_ne_params[0]:.2f}s, DELTA_T={max_ne_params[1]:.2f}s\n"
            report += f"   - 如果优先考虑系统效率，推荐: T_SAFE={max_payoff_params[0]:.2f}s, DELTA_T={max_payoff_params[1]:.2f}s\n"
    
    # 添加关于T_SAFE的分析
    t_safe_effect = {}
    for t_safe in t_safe_values:
        avg_ne_count = sum(results[t_safe][dt]['num_nash_equilibria'] for dt in delta_t_values) / len(delta_t_values)
        t_safe_effect[t_safe] = avg_ne_count
    
    max_t_safe = max(t_safe_effect.items(), key=lambda x: x[1])[0]
    min_t_safe = min(t_safe_effect.items(), key=lambda x: x[1])[0]
    
    report += f"   - T_SAFE参数分析: 当T_SAFE={max_t_safe:.2f}s时，平均纳什均衡数量最多；当T_SAFE={min_t_safe:.2f}s时，最少。\n"
    
    # 添加关于DELTA_T的分析
    delta_t_effect = {}
    for delta_t in delta_t_values:
        avg_ne_count = sum(results[t_safe][delta_t]['num_nash_equilibria'] for t_safe in t_safe_values) / len(t_safe_values)
        delta_t_effect[delta_t] = avg_ne_count
    
    max_delta_t = max(delta_t_effect.items(), key=lambda x: x[1])[0]
    min_delta_t = min(delta_t_effect.items(), key=lambda x: x[1])[0]
    
    report += f"   - DELTA_T参数分析: 当DELTA_T={max_delta_t:.2f}s时，平均纳什均衡数量最多；当DELTA_T={min_delta_t:.2f}s时，最少。\n"
    
    # 总结
    report += "\n   总结: 参数选择应根据具体应用场景的需求进行权衡。较大的安全时间间隔提高了安全性但可能降低效率，"
    report += "而较大的时间调整步长增加了车辆的灵活性但可能导致更大的偏差。"
    
    return report

# --- 主执行流程 ---
if __name__ == "__main__":
    # 定义车辆配置
    vehicle_configs = [
        # 配置1: 标准车辆配置
        [
            Vehicle(id=1, d=100, v=10, a_max=2),
            Vehicle(id=2, d=80, v=12, a_max=2.5),
            Vehicle(id=3, d=120, v=11, a_max=1.8)
        ],
        # 配置2: 高速车辆配置
        [
            Vehicle(id=1, d=150, v=20, a_max=3),
            Vehicle(id=2, d=130, v=22, a_max=3.5),
            Vehicle(id=3, d=170, v=21, a_max=2.8)
        ]
    ]
    
    # 定义权重配置
    weight_configs = [
        # 配置1: 平衡权重
        {
            "alpha1": 0.5, # r_o
            "alpha2": 0.3, # r_e
            "alpha3": 0.2, # r_c
            "alpha4": 0.1  # r_s
        },
        # 配置2: 效率优先
        {
            "alpha1": 0.3, # r_o
            "alpha2": 0.5, # r_e
            "alpha3": 0.1, # r_c
            "alpha4": 0.1  # r_s
        }
    ]
    
    # 选择要使用的配置
    vehicles = vehicle_configs[0]  # 使用标准车辆配置
    alphas = weight_configs[0]     # 使用平衡权重配置
    
    # 定义参数范围
    t_safe_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    delta_t_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    
    # 运行参数敏感性分析
    print("正在进行参数敏感性分析...")
    results = analyze_nash_equilibria_for_parameters(vehicles, t_safe_values, delta_t_values, alphas)
    print("分析完成!")
    
    # 绘制综合稳定性分析图
    print("\n生成综合稳定性分析图...")
    plot_comprehensive_stability_analysis(results, t_safe_values, delta_t_values)
    
    # 绘制最优纳什均衡热力图
    print("\n生成最优纳什均衡热力图...")
    plot_heatmap_analysis(results, t_safe_values, delta_t_values)
    
    # 生成建议报告
    print("\n生成参数敏感性分析报告...")
    recommendations = generate_recommendations(results, t_safe_values, delta_t_values, vehicles)
    print(recommendations)