import math
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 游戏配置类
class GameConfig:
    def __init__(self, t_safe=1.5, delta_t=0.5, alphas=None):
        self.t_safe = t_safe  # 安全时间间隔 (s)
        self.delta_t = delta_t  # 时间调整步长 (s)
        
        # 支付函数权重
        if alphas is None:
            self.alphas = {
                "alpha1": 0.5,  # r_o: GLOSO偏差惩罚
                "alpha2": 0.3,  # r_e: 效率收益
                "alpha3": 0.2,  # r_c: 舒适性收益
                "alpha4": 0.1   # r_s: 安全收益
            }
        else:
            self.alphas = alphas

# 车辆状态类
class VehicleState:
    def __init__(self, id, d, v, a_max):
        self.id = id  # 原始ID
        self.d = d    # 到达冲突区域的距离 (m)
        self.v = v    # 当前速度 (m/s)
        self.a_max = a_max  # 最大加速度 (m/s^2)
        self.t_min = self.calculate_t_min()  # 以最大加速度到达的最短时间 (s)
        self.eta = d / v if v > 0 else float('inf')  # 预计到达时间 (s)
        
        # GLOSO 和 LOGA 阶段会填充这些值
        self.gloso_id = -1  # 在GLOSO排序后的ID
        self.t_seq_gloso = 0.0  # GLOSO建议的到达时间
        self.k_loga = 0  # LOGA阶段选择的调整因子
        self.t_tar_loga = 0.0  # LOGA阶段的目标到达时间

    def calculate_t_min(self):
        # 解方程 d = v*t + 0.5*a_max*t^2 for t
        a = 0.5 * self.a_max
        b = self.v
        c = -self.d
        
        if abs(a) < 1e-9:  # 处理a_max可能为0的情况
            if abs(b) < 1e-9:  # 如果速度和加速度都为0
                return float('inf') if c < 0 else (0 if c==0 else float('inf'))
            if b > 0:
                return -c / b if -c / b >=0 else float('inf')
            else:  # b < 0, 除非d=0, 否则无法到达
                return 0 if c == 0 else float('inf')

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return float('inf')  # 无实数解，意味着无法到达
        
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2*a)
        t2 = (-b - sqrt_discriminant) / (2*a)
        
        # 返回非负的最小时间
        valid_times = []
        if t1 >= -1e-9:  # 允许微小的负值，当作0
            valid_times.append(max(0, t1))
        if t2 >= -1e-9:
            valid_times.append(max(0, t2))
        
        return min(valid_times) if valid_times else float('inf')

    def __repr__(self):
        return (f"Veh(id={self.id}, d={self.d}, v={self.v}, a_max={self.a_max}, "
                f"t_min={self.t_min:.2f}s, eta={self.eta:.2f}s, "
                f"gloso_id={self.gloso_id}, t_seq_gloso={self.t_seq_gloso:.2f}s)")

# 游戏结果类
class GameResults:
    def __init__(self):
        self.gloso_order = []  # GLOSO阶段确定的车辆顺序
        self.t_seq_gloso = []  # GLOSO阶段确定的到达时间序列
        self.legal_strategies = {}  # 合法策略及其支付
        self.nash_equilibria = []  # 纳什均衡策略
        self.nash_t_tar_profiles = []  # 纳什均衡对应的目标到达时间
        self.nash_payoffs = []  # 纳什均衡对应的支付

# GLOSO-LOGA游戏类
class GLOSOLOGAGame:
    def __init__(self, config=None):
        self.config = config if config else GameConfig()
        self.vehicles = []  # 车辆列表
        self.results = GameResults()  # 游戏结果

    def create_vehicle(self, id, d, v, a_max):
        """创建并添加一个新车辆"""
        vehicle = VehicleState(id, d, v, a_max)
        self.vehicles.append(vehicle)
        return vehicle

    def run_gloso(self):
        """执行GLOSO阶段，确定排序和计划到达时间"""
        # 按ETA排序
        sorted_vehicles = sorted(self.vehicles, key=lambda veh: veh.eta)
        
        for i, veh in enumerate(sorted_vehicles):
            veh.gloso_id = i + 1

        t_seq_gloso_list = []
        for i, veh in enumerate(sorted_vehicles):
            if i == 0: 
                veh.t_seq_gloso = max(veh.eta, veh.t_min)
            else:
                prev_veh_t_seq = sorted_vehicles[i-1].t_seq_gloso
                veh.t_seq_gloso = max(prev_veh_t_seq + self.config.t_safe, veh.eta, veh.t_min)
            t_seq_gloso_list.append(veh.t_seq_gloso)
        
        # 保存结果
        self.results.gloso_order = sorted_vehicles
        self.results.t_seq_gloso = t_seq_gloso_list
        
        return sorted_vehicles, t_seq_gloso_list

    def calculate_individual_payoff(self, veh_idx, t_tar_profile, k_profile):
        """计算单个车辆的支付函数"""
        veh = self.results.gloso_order[veh_idx]
        t_tar_self = t_tar_profile[veh_idx]

        # 支付函数组成部分
        r_o = -(t_tar_self - veh.t_seq_gloso)**2  # GLOSO偏差惩罚
        r_e = -t_tar_self  # 效率收益
        r_c = -(t_tar_self - veh.eta)**2  # 舒适性收益
        r_s = 0.0  # 安全收益（简化版本）

        # 计算总支付
        payoff = (self.config.alphas["alpha1"] * r_o +
                  self.config.alphas["alpha2"] * r_e +
                  self.config.alphas["alpha3"] * r_c +
                  self.config.alphas["alpha4"] * r_s)
        return payoff

    def find_nash_equilibria(self):
        """寻找纯策略纳什均衡"""
        gloso_vehicles = self.results.gloso_order
        legal_payoffs = self.results.legal_strategies
        nash_equilibria = []
        possible_k_values = [-1, 0, 1]  # 可能的k值

        for k_profile_star, payoffs_star in legal_payoffs.items():
            is_ne = True
            for player_idx in range(len(gloso_vehicles)): 
                current_payoff_for_player = payoffs_star[player_idx]
                
                # 检查玩家是否有更好的单边偏离策略
                for k_alternative in possible_k_values:
                    if k_alternative == k_profile_star[player_idx]:
                        continue  # 跳过当前策略

                    # 构造偏离策略
                    k_profile_deviated_list = list(k_profile_star)
                    k_profile_deviated_list[player_idx] = k_alternative
                    k_profile_deviated = tuple(k_profile_deviated_list)

                    # 检查偏离策略是否合法且是否提高支付
                    if k_profile_deviated in legal_payoffs:
                        payoffs_deviated = legal_payoffs[k_profile_deviated]
                        payoff_after_deviation_for_player = payoffs_deviated[player_idx]
                        if payoff_after_deviation_for_player > current_payoff_for_player + 1e-9:  # 允许微小误差比较
                            is_ne = False
                            break 
                if not is_ne:
                    break 
            
            if is_ne:
                nash_equilibria.append(k_profile_star)
        
        # 保存纳什均衡结果
        self.results.nash_equilibria = nash_equilibria
        
        # 计算纳什均衡对应的目标到达时间和支付
        nash_t_tar_profiles = []
        nash_payoffs = []
        for ne in nash_equilibria:
            t_tar_profile = self._calculate_t_tar_profile(ne)
            nash_t_tar_profiles.append(t_tar_profile)
            nash_payoffs.append(legal_payoffs[ne])
        
        self.results.nash_t_tar_profiles = nash_t_tar_profiles
        self.results.nash_payoffs = nash_payoffs
        
        return nash_equilibria

    def _calculate_t_tar_profile(self, k_profile):
        """计算给定k_profile的目标到达时间"""
        t_tar_profile = []
        for i, veh in enumerate(self.results.gloso_order):
            k_val = k_profile[i]
            t_tar = veh.t_seq_gloso + k_val * self.config.delta_t
            t_tar_profile.append(t_tar)
        return tuple(t_tar_profile)

    def _is_profile_legal(self, t_tar_profile):
        """检查目标到达时间是否合法"""
        gloso_vehicles = self.results.gloso_order
        
        # 检查每辆车的目标到达时间是否不小于其最小到达时间
        for i in range(len(gloso_vehicles)):
            if t_tar_profile[i] < gloso_vehicles[i].t_min - 1e-6: 
                return False
        
        # 检查相邻车辆之间的安全时间间隔
        for i in range(1, len(gloso_vehicles)):
            if t_tar_profile[i] - t_tar_profile[i-1] < self.config.t_safe - 1e-6:
                return False
        
        return True

    def run_complete_game(self):
        """运行完整的GLOSO-LOGA游戏"""
        # 1. 运行GLOSO阶段
        self.run_gloso()
        
        # 2. 生成所有可能的k值组合
        k_values = [-1, 0, 1]
        all_k_profiles = list(product(k_values, repeat=len(self.results.gloso_order)))
        legal_payoffs = {}
        
        # 3. 计算合法策略及其支付
        for k_profile in all_k_profiles:
            t_tar_profile = self._calculate_t_tar_profile(k_profile)
            is_legal = self._is_profile_legal(t_tar_profile)
            
            if is_legal:
                payoffs = []
                for i in range(len(self.results.gloso_order)):
                    payoff_veh_i = self.calculate_individual_payoff(
                        veh_idx=i,
                        t_tar_profile=t_tar_profile,
                        k_profile=k_profile
                    )
                    payoffs.append(payoff_veh_i)
                legal_payoffs[k_profile] = tuple(payoffs)
        
        self.results.legal_strategies = legal_payoffs
        
        # 4. 寻找纳什均衡
        self.find_nash_equilibria()
        
        return self.results

    def print_detailed_results(self):
        """打印详细的游戏结果"""
        results = self.results
        
        print("\n=== GLOSO-LOGA 游戏结果 ===\n")
        
        # 1. GLOSO阶段结果
        print("--- GLOSO 阶段结果 ---")
        print("GLOSO 通行顺序 (按原始ID):", [veh.id for veh in results.gloso_order])
        print("GLOSO 计划到达时间序列 T_seq*:", [f"{t:.2f}s" for t in results.t_seq_gloso])
        print("详细车辆GLOSO信息:")
        for veh in results.gloso_order:
            print(veh)
        print("-" * 50)
        
        # 2. 合法策略统计
        print(f"总共 {3**len(results.gloso_order)} 种策略组合, 其中 {len(results.legal_strategies)} 种是合法的。")
        
        # 3. 纳什均衡结果
        print("\n--- 纳什均衡结果 ---")
        if not results.nash_equilibria:
            print("没有找到纯策略纳什均衡。")
        else:
            print(f"找到了 {len(results.nash_equilibria)} 个纯策略纳什均衡:")
            print(f"{'NE k-Profile':<15} | {'t_tar Profile (s)':<25} | {'Payoffs'}")
            print("-" * 70)
            for i, ne_k_profile in enumerate(results.nash_equilibria):
                ne_t_tar_profile = results.nash_t_tar_profiles[i]
                ne_payoffs = results.nash_payoffs[i]
                
                ne_t_tar_str = ", ".join([f"{t:.2f}" for t in ne_t_tar_profile])
                ne_payoffs_str = ", ".join([f"{p:.3f}" for p in ne_payoffs])
                print(f"{str(ne_k_profile):<15} | ({ne_t_tar_str}) | ({ne_payoffs_str})")
        print("-" * 50)

    def visualize_results(self, figsize=(18, 12)):
        """可视化游戏结果"""
        if not self.results.gloso_order:
            print("没有可视化的结果。请先运行游戏。")
            return
        
        # 创建子图布局
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # 1. 时间线比较
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_timeline_comparison(ax1)
        
        # 2. 支付热力图
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_payoff_heatmap(ax2)
        
        # 3. 策略分析
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_strategy_analysis(ax3)
        
        # 4. 车辆状态比较
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_vehicle_comparison(ax4)
        
        # 5. 纳什均衡比较
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_nash_comparison(ax5)
        
        # 6. 安全分析
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_safety_analysis(ax6)
        
        plt.tight_layout()
        plt.savefig("gloso_loga_game_results.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_timeline_comparison(self, ax):
        """绘制时间线比较图"""
        vehicles = self.results.gloso_order
        
        # 收集数据
        vehicle_ids = [f"车辆{v.id}" for v in vehicles]
        etas = [v.eta for v in vehicles]
        t_mins = [v.t_min for v in vehicles]
        t_seq_glosos = [v.t_seq_gloso for v in vehicles]
        
        # 如果有纳什均衡，添加第一个纳什均衡的目标到达时间
        t_tar_nash = None
        if self.results.nash_t_tar_profiles:
            t_tar_nash = self.results.nash_t_tar_profiles[0]
        
        # 设置x轴位置
        x = np.arange(len(vehicle_ids))
        width = 0.2
        
        # 绘制条形图
        ax.bar(x - 1.5*width, etas, width, label='ETA', color='skyblue')
        ax.bar(x - 0.5*width, t_mins, width, label='最小到达时间', color='lightgreen')
        ax.bar(x + 0.5*width, t_seq_glosos, width, label='GLOSO时间', color='salmon')
        
        if t_tar_nash:
            ax.bar(x + 1.5*width, t_tar_nash, width, label='纳什均衡时间', color='purple')
        
        # 添加标签和图例
        ax.set_xlabel('车辆')
        ax.set_ylabel('时间 (s)')
        ax.set_title('车辆到达时间比较')
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_ids)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_payoff_heatmap(self, ax):
        """绘制支付热力图"""
        if not self.results.nash_equilibria or len(self.results.gloso_order) != 3:
            ax.text(0.5, 0.5, '需要3辆车和至少1个纳什均衡\n才能生成支付热力图', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # 为了简化，我们只考虑第一个车辆的支付，并假设其他两辆车采用纳什均衡策略
        vehicle_idx = 0
        ne = self.results.nash_equilibria[0]
        
        # 创建k值网格
        k_values = [-1, 0, 1]
        k2, k3 = ne[1], ne[2]  # 固定车辆2和3的策略为纳什均衡策略
        
        # 计算支付矩阵
        payoff_matrix = np.zeros((3, 3))  # 3x3矩阵，对应车辆1的3种策略和车辆2的3种策略
        
        for i, k1 in enumerate(k_values):
            for j, k2_alt in enumerate(k_values):
                k_profile = (k1, k2_alt, k3)
                t_tar_profile = self._calculate_t_tar_profile(k_profile)
                
                if self._is_profile_legal(t_tar_profile):
                    payoff = self.calculate_individual_payoff(vehicle_idx, t_tar_profile, k_profile)
                    payoff_matrix[i, j] = payoff
                else:
                    payoff_matrix[i, j] = float('nan')  # 非法策略
        
        # 绘制热力图
        im = ax.imshow(payoff_matrix, cmap='viridis')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('车辆1的支付值')
        
        # 添加标签
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(k_values)
        ax.set_yticklabels(k_values)
        ax.set_xlabel('车辆2的k值')
        ax.set_ylabel('车辆1的k值')
        ax.set_title('车辆1的支付热力图 (车辆3的k值固定为{})'.format(k3))
        
        # 在每个单元格中添加支付值
        for i in range(3):
            for j in range(3):
                if not np.isnan(payoff_matrix[i, j]):
                    text = ax.text(j, i, f"{payoff_matrix[i, j]:.2f}",
                                ha="center", va="center", color="w" if payoff_matrix[i, j] < -1 else "black")

    def _plot_strategy_analysis(self, ax):
        """绘制策略分析图"""
        if not self.results.legal_strategies:
            ax.text(0.5, 0.5, '没有合法策略可供分析', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # 统计每个k值在合法策略中的出现频率
        k_counts = {-1: [0] * len(self.results.gloso_order), 
                    0: [0] * len(self.results.gloso_order), 
                    1: [0] * len(self.results.gloso_order)}
        
        for k_profile in self.results.legal_strategies.keys():
            for i, k in enumerate(k_profile):
                k_counts[k][i] += 1
        
        # 转换为百分比
        total_legal = len(self.results.legal_strategies)
        for k in k_counts:
            for i in range(len(k_counts[k])):
                k_counts[k][i] = (k_counts[k][i] / total_legal) * 100
        
        # 设置x轴位置
        vehicle_ids = [f"车辆{v.id}" for v in self.results.gloso_order]
        x = np.arange(len(vehicle_ids))
        width = 0.25
        
        # 绘制条形图
        ax.bar(x - width, k_counts[-1], width, label='k=-1', color='#ff9999')
        ax.bar(x, k_counts[0], width, label='k=0', color='#66b3ff')
        ax.bar(x + width, k_counts[1], width, label='k=1', color='#99ff99')
        
        # 添加标签和图例
        ax.set_xlabel('车辆')
        ax.set_ylabel('在合法策略中的出现频率 (%)')
        ax.set_title('各车辆k值在合法策略中的分布')
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_ids)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_vehicle_comparison(self, ax):
        """绘制车辆状态比较图"""
        vehicles = self.results.gloso_order
        
        # 收集数据
        vehicle_ids = [f"车辆{v.id}" for v in vehicles]
        distances = [v.d for v in vehicles]
        speeds = [v.v for v in vehicles]
        accel_max = [v.a_max for v in vehicles]
        
        # 创建多个y轴
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # 设置x轴位置
        x = np.arange(len(vehicle_ids))
        
        # 绘制线图
        ax.plot(x, distances, 'o-', color='blue', label='距离 (m)')
        ax2.plot(x, speeds, 's-', color='red', label='速度 (m/s)')
        ax3.plot(x, accel_max, '^-', color='green', label='最大加速度 (m/s²)')
        
        # 添加标签和图例
        ax.set_xlabel('车辆')
        ax.set_ylabel('距离 (m)', color='blue')
        ax2.set_ylabel('速度 (m/s)', color='red')
        ax3.set_ylabel('最大加速度 (m/s²)', color='green')
        ax.set_title('车辆状态比较')
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_ids)
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')
        
        ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_nash_comparison(self, ax):
        """绘制纳什均衡比较图"""
        if not self.results.nash_equilibria:
            ax.text(0.5, 0.5, '没有找到纳什均衡', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # 收集数据
        ne_labels = [str(ne) for ne in self.results.nash_equilibria]
        total_payoffs = [sum(payoffs) for payoffs in self.results.nash_payoffs]
        individual_payoffs = [list(payoffs) for payoffs in self.results.nash_payoffs]
        
        # 设置x轴位置
        x = np.arange(len(ne_labels))
        width = 0.15
        
        # 绘制总支付条形图
        ax.bar(x, total_payoffs, width*3, label='总支付', color='purple', alpha=0.3)
        
        # 绘制个体支付条形图
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        for i in range(len(self.results.gloso_order)):
            vehicle_payoffs = [payoffs[i] for payoffs in individual_payoffs]
            offset = width * (i - (len(self.results.gloso_order) - 1) / 2)
            ax.bar(x + offset, vehicle_payoffs, width, 
                   label=f'车辆{self.results.gloso_order[i].id}', 
                   color=colors[i % len(colors)])
        
        # 添加标签和图例
        ax.set_xlabel('纳什均衡策略')
        ax.set_ylabel('支付值')
        ax.set_title('纳什均衡支付比较')
        ax.set_xticks(x)
        ax.set_xticklabels(ne_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_safety_analysis(self, ax):
        """绘制安全分析图"""
        if not self.results.nash_t_tar_profiles:
            ax.text(0.5, 0.5, '没有纳什均衡可供安全分析', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # 使用第一个纳什均衡进行分析
        t_tar_profile = self.results.nash_t_tar_profiles[0]
        
        # 计算相邻车辆之间的安全时间间隔
        safety_margins = []
        for i in range(1, len(t_tar_profile)):
            margin = t_tar_profile[i] - t_tar_profile[i-1]
            safety_margins.append(margin)
        
        # 设置x轴位置和标签
        vehicle_pairs = []
        for i in range(1, len(self.results.gloso_order)):
            v1 = self.results.gloso_order[i-1].id
            v2 = self.results.gloso_order[i].id
            vehicle_pairs.append(f"车辆{v1}-车辆{v2}")
        
        x = np.arange(len(vehicle_pairs))
        
        # 根据安全裕度设置颜色
        colors = []
        for margin in safety_margins:
            if margin < self.config.t_safe:
                colors.append('red')  # 不安全
            elif margin < self.config.t_safe * 1.5:
                colors.append('orange')  # 临界安全
            else:
                colors.append('green')  # 安全
        
        # 绘制条形图
        bars = ax.bar(x, safety_margins, color=colors)
        
        # 添加安全阈值线
        ax.axhline(y=self.config.t_safe, color='r', linestyle='--', 
                   label=f'安全阈值 ({self.config.t_safe}s)')
        
        # 添加标签和图例
        ax.set_xlabel('车辆对')
        ax.set_ylabel('时间间隔 (s)')
        ax.set_title('车辆间安全时间间隔分析')
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_pairs, rotation=45, ha='right')
        ax.legend()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{safety_margins[i]:.2f}s',
                    ha='center', va='bottom', rotation=0)
        
        ax.grid(True, linestyle='--', alpha=0.7)

# 创建示例场景
def create_example_scenario():
    game = GLOSOLOGAGame()
    game.create_vehicle(id=1, d=100, v=10, a_max=2)
    game.create_vehicle(id=2, d=80, v=12, a_max=2.5)
    game.create_vehicle(id=3, d=120, v=11, a_max=1.8)
    return game

# 运行示例
def run_example():
    game = create_example_scenario()
    game.run_complete_game()
    game.print_detailed_results()
    game.visualize_results()

# 主程序
if __name__ == "__main__":
    run_example()