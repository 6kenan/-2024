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
    return payoff, (r_o, r_e, r_c, r_s)  # 返回总支付和各组成部分

def find_nash_equilibria(legal_payoffs, gloso_vehicles):
    nash_equilibria = []
    possible_k_values = [-1, 0, 1]

    for k_profile_star, payoffs_star in legal_payoffs.items():
        is_ne = True
        for player_idx in range(len(gloso_vehicles)): 
            current_payoff_for_player = payoffs_star[player_idx][0]  # 总支付
            
            for k_alternative in possible_k_values:
                if k_alternative == k_profile_star[player_idx]:
                    continue 

                k_profile_deviated_list = list(k_profile_star)
                k_profile_deviated_list[player_idx] = k_alternative
                k_profile_deviated = tuple(k_profile_deviated_list)

                if k_profile_deviated in legal_payoffs:
                    payoffs_deviated = legal_payoffs[k_profile_deviated]
                    payoff_after_deviation_for_player = payoffs_deviated[player_idx][0]  # 总支付
                    if payoff_after_deviation_for_player > current_payoff_for_player + 1e-9: # 允许微小误差比较
                        is_ne = False
                        break 
            if not is_ne:
                break 
        
        if is_ne:
            nash_equilibria.append(k_profile_star)
            
    return nash_equilibria

# --- 增强可视化函数 ---
def plot_vehicle_timeline(gloso_vehicles, nash_equilibria, legal_payoffs):
    """
    绘制车辆时间线比较图，比较ETA、GLOSO和纳什均衡的到达时间
    """
    if not nash_equilibria:
        print("没有纳什均衡可供可视化。")
        return
    
    # 选择第一个纳什均衡进行可视化
    ne_k_profile = nash_equilibria[0]
    ne_t_tar_profile = calculate_t_tar_profile(gloso_vehicles, ne_k_profile)
    
    # 收集数据
    vehicle_ids = [f"车辆{v.id}" for v in gloso_vehicles]
    etas = [v.eta for v in gloso_vehicles]
    t_seq_glosos = [v.t_seq_gloso for v in gloso_vehicles]
    t_tar_nash = list(ne_t_tar_profile)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置x轴位置
    x = np.arange(len(vehicle_ids))
    width = 0.25
    
    # 绘制条形图
    ax.bar(x - width, etas, width, label='ETA', color='skyblue')
    ax.bar(x, t_seq_glosos, width, label='GLOSO时间', color='salmon')
    ax.bar(x + width, t_tar_nash, width, label='纳什均衡时间', color='purple')
    
    # 添加标签和图例
    ax.set_xlabel('车辆')
    ax.set_ylabel('时间 (s)')
    ax.set_title('车辆到达时间比较: ETA vs GLOSO vs 纳什均衡')
    ax.set_xticks(x)
    ax.set_xticklabels(vehicle_ids)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(etas):
        ax.text(i - width, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(t_seq_glosos):
        ax.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(t_tar_nash):
        ax.text(i + width, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("vehicle_timeline_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_payoff_heatmap(gloso_vehicles, legal_payoffs):
    """
    绘制支付热力图，展示不同策略组合下的支付情况
    """
    if len(gloso_vehicles) != 3:
        print("支付热力图需要3辆车。")
        return
    
    # 创建一个3x3的子图布局，每个子图对应一个车辆在不同k3值下的支付
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('车辆支付热力图分析', fontsize=16)
    
    k_values = [-1, 0, 1]
    
    # 为每个k3值创建一个子图
    for k3_idx, k3 in enumerate(k_values):
        # 创建支付矩阵 (k1 x k2)
        payoff_matrices = [np.zeros((3, 3)) for _ in range(3)]  # 3个矩阵，对应3辆车
        
        # 填充支付矩阵
        for k1_idx, k1 in enumerate(k_values):
            for k2_idx, k2 in enumerate(k_values):
                k_profile = (k1, k2, k3)
                if k_profile in legal_payoffs:
                    payoffs = legal_payoffs[k_profile]
                    for veh_idx in range(3):
                        payoff_matrices[veh_idx][k1_idx, k2_idx] = payoffs[veh_idx][0]  # 总支付
                else:
                    for veh_idx in range(3):
                        payoff_matrices[veh_idx][k1_idx, k2_idx] = float('nan')  # 非法策略
        
        # 绘制热力图
        for veh_idx in range(3):
            ax = axes[veh_idx, k3_idx]
            im = ax.imshow(payoff_matrices[veh_idx], cmap='viridis', origin='lower')
            
            # 添加标签
            if veh_idx == 2:  # 最后一行
                ax.set_xlabel(f'车辆2的k值 (k3={k3})')
            if k3_idx == 0:  # 第一列
                ax.set_ylabel(f'车辆1的k值 (车辆{veh_idx+1}的支付)')
            
            # 设置刻度
            ax.set_xticks(np.arange(3))
            ax.set_yticks(np.arange(3))
            ax.set_xticklabels(k_values)
            ax.set_yticklabels(k_values)
            
            # 在每个单元格中添加支付值
            for i in range(3):
                for j in range(3):
                    if not np.isnan(payoff_matrices[veh_idx][i, j]):
                        text = ax.text(j, i, f"{payoff_matrices[veh_idx][i, j]:.2f}",
                                    ha="center", va="center", 
                                    color="w" if payoff_matrices[veh_idx][i, j] < -1 else "black",
                                    fontsize=8)
            
            # 添加颜色条
            if veh_idx == 0 and k3_idx == 2:  # 只在右上角添加一个颜色条
                cbar = fig.colorbar(im, ax=axes[0, 2], shrink=0.8)
                cbar.set_label('支付值')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
    plt.savefig("payoff_heatmap_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_payoff_components_analysis(gloso_vehicles, nash_equilibria, legal_payoffs):
    """
    分析纳什均衡中各车辆支付函数的组成部分
    """
    if not nash_equilibria:
        print("没有纳什均衡可供分析。")
        return
    
    # 收集数据
    ne_labels = [str(ne) for ne in nash_equilibria]
    component_names = ['GLOSO偏差 (r_o)', '效率 (r_e)', '舒适性 (r_c)', '安全 (r_s)']  
    component_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    # 创建图形
    fig, axes = plt.subplots(len(gloso_vehicles), 1, figsize=(12, 4*len(gloso_vehicles)))
    if len(gloso_vehicles) == 1:
        axes = [axes]  # 确保axes是列表
    
    # 为每辆车创建一个子图
    for veh_idx, veh in enumerate(gloso_vehicles):
        ax = axes[veh_idx]
        
        # 收集该车辆在各纳什均衡下的支付组成部分
        components_data = []
        for ne in nash_equilibria:
            payoff_data = legal_payoffs[ne][veh_idx][1]  # 支付组成部分
            components_data.append(payoff_data)
        
        # 转换为numpy数组以便绘图
        components_array = np.array(components_data)
        
        # 设置x轴位置
        x = np.arange(len(ne_labels))
        width = 0.2
        
        # 绘制条形图
        bottom = np.zeros(len(ne_labels))
        for i, (component, color) in enumerate(zip(components_array.T, component_colors)):
            weighted_component = component * list(ALPHAS.values())[i]  # 应用权重
            ax.bar(x, weighted_component, width, bottom=bottom, label=component_names[i], color=color)
            
            # 在每个部分上添加数值标签
            for j, v in enumerate(weighted_component):
                if abs(v) > 0.01:  # 只显示非零值
                    ax.text(j, bottom[j] + v/2, f'{v:.2f}', ha='center', va='center', fontsize=8, color='black')
            
            bottom += weighted_component
        
        # 添加总支付值标签
        for j, total in enumerate(bottom):
            ax.text(j, total + 0.1, f'总计: {total:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加标签和图例
        ax.set_xlabel('纳什均衡策略')
        ax.set_ylabel('加权支付值')
        ax.set_title(f'车辆{veh.id}的支付函数组成分析')
        ax.set_xticks(x)
        ax.set_xticklabels(ne_labels)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("payoff_components_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_sensitivity_analysis(initial_vehicles, t_safe_values=None):
    """
    分析T_SAFE参数对纳什均衡数量的影响
    """
    if t_safe_values is None:
        t_safe_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    # 收集数据
    nash_counts = []
    legal_counts = []
    
    for t_safe in t_safe_values:
        # 复制车辆以避免修改原始数据
        vehicles_copy = [Vehicle(v.id, v.d, v.v, v.a_max) for v in initial_vehicles]
        
        # 临时修改全局参数
        global T_SAFE
        original_t_safe = T_SAFE
        T_SAFE = t_safe
        
        # 运行GLOSO阶段
        gloso_ordered_vehicles, _ = run_gloso(vehicles_copy)
        
        # 生成所有可能的k值组合
        k_values = [-1, 0, 1]
        all_k_profiles = list(product(k_values, repeat=len(gloso_ordered_vehicles)))
        
        # 计算合法策略及其支付
        legal_payoffs = {}
        for k_profile in all_k_profiles:
            t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, k_profile)
            is_legal = is_profile_legal(t_tar_profile, gloso_ordered_vehicles)
            
            if is_legal:
                payoffs = []
                for i in range(len(gloso_ordered_vehicles)):
                    payoff_veh_i, components = calculate_payoff_for_vehicle(
                        veh_idx_in_gloso=i,
                        t_tar_profile=t_tar_profile,
                        k_profile=k_profile,
                        gloso_vehicles=gloso_ordered_vehicles
                    )
                    payoffs.append((payoff_veh_i, components))
                legal_payoffs[k_profile] = tuple(payoffs)
        
        # 寻找纳什均衡
        nash_equilibria = find_nash_equilibria(legal_payoffs, gloso_ordered_vehicles)
        
        # 记录结果
        nash_counts.append(len(nash_equilibria))
        legal_counts.append(len(legal_payoffs))
        
        # 恢复原始参数
        T_SAFE = original_t_safe
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制纳什均衡数量曲线
    color = 'tab:blue'
    ax1.set_xlabel('安全时间间隔 (T_SAFE)')
    ax1.set_ylabel('纳什均衡数量', color=color)
    ax1.plot(t_safe_values, nash_counts, 'o-', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建第二个y轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('合法策略数量', color=color)
    ax2.plot(t_safe_values, legal_counts, 's--', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加标题和网格
    plt.title('T_SAFE参数敏感性分析')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(nash_counts):
        ax1.text(t_safe_values[i], v + 0.1, str(v), ha='center', va='bottom', color='tab:blue')
    for i, v in enumerate(legal_counts):
        ax2.text(t_safe_values[i], v + 0.1, str(v), ha='center', va='bottom', color='tab:red')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, ['纳什均衡数量', '合法策略数量'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig("t_safe_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_comparison(gloso_vehicles, nash_equilibria, legal_payoffs):
    """
    综合比较分析，包括纳什均衡支付比较、延迟分析和系统效率比较
    """
    if not nash_equilibria:
        print("没有纳什均衡可供分析。")
        return
    
    # 创建一个3x1的子图布局
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. 纳什均衡支付比较
    ax1 = axes[0]
    
    # 收集数据
    ne_labels = [str(ne) for ne in nash_equilibria]
    total_payoffs = []
    individual_payoffs = []
    
    for ne in nash_equilibria:
        payoffs = legal_payoffs[ne]
        total_payoff = sum(p[0] for p in payoffs)  # 总支付
        total_payoffs.append(total_payoff)
        individual_payoffs.append([p[0] for p in payoffs])  # 个体支付
    
    # 设置x轴位置
    x = np.arange(len(ne_labels))
    width = 0.15
    
    # 绘制总支付条形图
    ax1.bar(x, total_payoffs, width*3, label='总支付', color='purple', alpha=0.3)
    
    # 绘制个体支付条形图
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for i in range(len(gloso_vehicles)):
        vehicle_payoffs = [payoffs[i] for payoffs in individual_payoffs]
        offset = width * (i - (len(gloso_vehicles) - 1) / 2)
        ax1.bar(x + offset, vehicle_payoffs, width, 
               label=f'车辆{gloso_vehicles[i].id}', 
               color=colors[i % len(colors)])
    
    # 添加标签和图例
    ax1.set_xlabel('纳什均衡策略')
    ax1.set_ylabel('支付值')
    ax1.set_title('纳什均衡支付比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ne_labels)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 延迟分析
    ax2 = axes[1]
    
    # 计算各阶段的延迟
    eta_to_gloso_delays = []
    gloso_to_nash_delays = []
    total_delays = []
    
    for i, ne in enumerate(nash_equilibria):
        ne_t_tar_profile = calculate_t_tar_profile(gloso_vehicles, ne)
        
        eta_to_gloso_delay = sum(v.t_seq_gloso - v.eta for v in gloso_vehicles)
        gloso_to_nash_delay = sum(ne_t_tar_profile[j] - gloso_vehicles[j].t_seq_gloso for j in range(len(gloso_vehicles)))
        total_delay = eta_to_gloso_delay + gloso_to_nash_delay
        
        eta_to_gloso_delays.append(eta_to_gloso_delay)
        gloso_to_nash_delays.append(gloso_to_nash_delay)
        total_delays.append(total_delay)
    
    # 设置x轴位置
    x = np.arange(len(ne_labels))
    width = 0.25
    
    # 绘制条形图
    ax2.bar(x - width, eta_to_gloso_delays, width, label='ETA到GLOSO延迟', color='skyblue')
    ax2.bar(x, gloso_to_nash_delays, width, label='GLOSO到纳什均衡延迟', color='salmon')
    ax2.bar(x + width, total_delays, width, label='总延迟', color='purple')
    
    # 添加标签和图例
    ax2.set_xlabel('纳什均衡策略')
    ax2.set_ylabel('延迟时间 (s)')
    ax2.set_title('各阶段延迟分析')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ne_labels)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 系统效率比较
    ax3 = axes[2]
    
    # 计算各阶段的系统通行时间
    eta_total_times = []
    gloso_total_times = []
    nash_total_times = []
    
    for i, ne in enumerate(nash_equilibria):
        ne_t_tar_profile = calculate_t_tar_profile(gloso_vehicles, ne)
        
        # 计算最后一辆车通过的时间
        eta_total_time = max(v.eta for v in gloso_vehicles)
        gloso_total_time = max(v.t_seq_gloso for v in gloso_vehicles)
        nash_total_time = max(ne_t_tar_profile)
        
        eta_total_times.append(eta_total_time)
        gloso_total_times.append(gloso_total_time)
        nash_total_times.append(nash_total_time)
    
    # 设置x轴位置
    x = np.arange(len(ne_labels))
    width = 0.25
    
    # 绘制条形图
    ax3.bar(x - width, eta_total_times, width, label='ETA总通行时间', color='skyblue')
    ax3.bar(x, gloso_total_times, width, label='GLOSO总通行时间', color='salmon')
    ax3.bar(x + width, nash_total_times, width, label='纳什均衡总通行时间', color='purple')
    
    # 添加标签和图例
    ax3.set_xlabel('纳什均衡策略')
    ax3.set_ylabel('总通行时间 (s)')
    ax3.set_title('系统效率比较')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ne_labels)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# --- 主执行流程 ---
if __name__ == "__main__":
    # 初始化车辆数据
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

    # 运行GLOSO阶段
    gloso_ordered_vehicles, t_seq_gloso_values = run_gloso(initial_vehicles_data)
    
    print("--- GLOSO 阶段结果 ---")
    print("GLOSO 通行顺序 (按原始ID):", [veh.id for veh in gloso_ordered_vehicles])
    print("GLOSO 计划到达时间序列 T_seq*:", [f"{t:.2f}s" for t in t_seq_gloso_values])
    print("详细车辆GLOSO信息:")
    for veh in gloso_ordered_vehicles:
        print(veh)
    print("-" * 50)

    # 运行LOGA阶段
    k_values = [-1, 0, 1]
    all_k_profiles = list(product(k_values, repeat=len(gloso_ordered_vehicles)))
    legal_payoffs = {} 
    
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
                payoff_veh_i, components = calculate_payoff_for_vehicle(
                    veh_idx_in_gloso=i,
                    t_tar_profile=t_tar_profile,
                    k_profile=k_profile,
                    gloso_vehicles=gloso_ordered_vehicles
                )
                payoffs.append((payoff_veh_i, components))
            legal_payoffs[k_profile] = tuple(payoffs)
            payoffs_for_profile_str = f"({payoffs[0][0]:.3f}, {payoffs[1][0]:.3f}, {payoffs[2][0]:.3f})"

        t_tar_profile_str = f"({t_tar_profile[0]:.2f}, {t_tar_profile[1]:.2f}, {t_tar_profile[2]:.2f})"
        print(f"{str(k_profile):<12} | {t_tar_profile_str:<25} | {str(is_legal):<8} | {payoffs_for_profile_str:<30}")

    print("-" * 80)
    print(f"总共 {len(all_k_profiles)} 种策略组合, 其中 {len(legal_payoffs)} 种是合法的。")
    print("-" * 50)
    
    # 寻找纳什均衡
    nash_equilibria = find_nash_equilibria(legal_payoffs, gloso_ordered_vehicles)
    
    print("--- 纳什均衡结果 ---")
    if not nash_equilibria:
        print("没有找到纯策略纳什均衡。")
    else:
        print(f"找到了 {len(nash_equilibria)} 个纯策略纳什均衡:")
        print(f"{'NE k-Profile':<15} | {'t_tar Profile (s)':<25} | {'Payoffs (R1, R2, R3)'}")
        print("-" * 70)
        for ne_k_profile in nash_equilibria:
            ne_t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, ne_k_profile)
            ne_payoffs = legal_payoffs[ne_k_profile]
            
            ne_t_tar_str = f"({ne_t_tar_profile[0]:.2f}, {ne_t_tar_profile[1]:.2f}, {ne_t_tar_profile[2]:.2f})"
            ne_payoffs_str = f"({ne_payoffs[0][0]:.3f}, {ne_payoffs[1][0]:.3f}, {ne_payoffs[2][0]:.3f})"
            print(f"{str(ne_k_profile):<15} | {ne_t_tar_str:<25} | {ne_payoffs_str}")
    print("-" * 50)
    
    # 运行增强可视化分析
    if nash_equilibria:
        print("\n--- 运行增强可视化分析 ---")
        
        # 1. 车辆时间线比较
        print("\n1. 绘制车辆时间线比较图...")
        plot_vehicle_timeline(gloso_ordered_vehicles, nash_equilibria, legal_payoffs)
        
        # 2. 支付热力图
        print("\n2. 绘制支付热力图...")
        plot_payoff_heatmap(gloso_ordered_vehicles, legal_payoffs)
        
        # 3. 支付组成部分分析
        print("\n3. 绘制支付组成部分分析图...")
        plot_payoff_components_analysis(gloso_ordered_vehicles, nash_equilibria, legal_payoffs)
        
        # 4. T_SAFE参数敏感性分析
        print("\n4. 绘制T_SAFE参数敏感性分析图...")
        plot_sensitivity_analysis(initial_vehicles_data)
        
        # 5. 综合比较分析
        print("\n5. 绘制综合比较分析图...")
        plot_comprehensive_comparison(gloso_ordered_vehicles, nash_equilibria, legal_payoffs)
    
    # 打印详细分析摘要
    print("\n=== 详细分析摘要 ===\n")
    print(f"车辆数量: {len(initial_vehicles_data)}")
    print(f"合法策略数量: {len(legal_payoffs)}")
    print(f"纳什均衡数量: {len(nash_equilibria)}")
    
    if nash_equilibria:
        # 计算系统总通行时间
        eta_total_time = max(v.eta for v in gloso_ordered_vehicles)
        gloso_total_time = max(v.t_seq_gloso for v in gloso_ordered_vehicles)
        
        ne_total_times = []
        for ne in nash_equilibria:
            ne_t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, ne)
            ne_total_times.append(max(ne_t_tar_profile))
        
        best_ne_idx = np.argmin(ne_total_times)
        best_ne_total_time = ne_total_times[best_ne_idx]
        best_ne = nash_equilibria[best_ne_idx]
        
        print(f"\n系统总通行时间:")
        print(f"  - ETA: {eta_total_time:.2f}s")
        print(f"  - GLOSO: {gloso_total_time:.2f}s (相比ETA延迟: {gloso_total_time-eta_total_time:.2f}s)")
        print(f"  - 最佳纳什均衡 {best_ne}: {best_ne_total_time:.2f}s (相比GLOSO延迟: {best_ne_total_time-gloso_total_time:.2f}s)")
        
        # 计算各车辆的时间变化
        best_ne_t_tar_profile = calculate_t_tar_profile(gloso_ordered_vehicles, best_ne)
        print(f"\n各车辆时间变化 (最佳纳什均衡 {best_ne}):")
        for i, veh in enumerate(gloso_ordered_vehicles):
            eta = veh.eta
            gloso = veh.t_seq_gloso
            nash = best_ne_t_tar_profile[i]
            print(f"  - 车辆{veh.id}: ETA={eta:.2f}s → GLOSO={gloso:.2f}s (Δ={gloso-eta:.2f}s) → Nash={nash:.2f}s (Δ={nash-gloso:.2f}s)")
    
    print("\n分析完成!")
