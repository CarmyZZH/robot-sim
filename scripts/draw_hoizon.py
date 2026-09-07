import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8, 4))

# 过去真实路径
t_past = np.linspace(0, 5, 50)
y_past = np.sin(0.5 * t_past)
ax.plot(t_past, y_past, color='k', lw=3, label="Past Execution (历史执行)")

# 未来系统参考线
t_future = np.linspace(5, 10, 50)
y_ref = np.sin(0.5 * t_future)
ax.plot(t_future, y_ref, 'k--', lw=2, label="Reference Trajectory (未来参考段)")

# N步前瞻预测 (预测多条可能，选出最优)
t_pred = np.linspace(5, 8, 30) # 前瞻窗口
for i in range(5):
    # 用一些二次曲线假装是寻优过程中的废弃解
    y_guess = y_past[-1] + (y_ref[15] - y_past[-1] + (i-2)*0.2)*np.linspace(0,1,30) - (i-2)*0.1*np.linspace(0,1,30)**2
    ax.plot(t_pred, y_guess, 'gray', alpha=0.3, lw=1)

# 最优预测轨迹 (MPC算出来的)
y_opt = y_past[-1] + (y_ref[15] - y_past[-1])*np.linspace(0,1,30) # 逐渐收敛
ax.plot(t_pred, y_opt, 'r-', lw=2.5, label="Optimal Prediction (最优预测窗口)")

# 第1步实际下发的指令线段
ax.plot(t_pred[:5], y_opt[:5], 'b-', lw=5, label="Executed Control (仅执行第一步)")

# 画出当前时刻的竖线
ax.axvline(x=5, color='gray', linestyle='-.')
ax.text(5.1, -0.8, "Current Time $k$\n(当前重求解时刻)", fontsize=11, color='dimgray')

# 时间窗口阴影区域
ax.axvspan(5, 8, color='red', alpha=0.08)
ax.text(6.5, -0.6, "Predictive Horizon $N=10$\n(预测时域)", fontsize=11, color='red', ha='center')

ax.set_title("Receding Horizon Control (滚动时域机理)", fontweight='bold')
ax.set_xlabel("Time Step (k)")
ax.set_ylabel("State (e.g. Velocity/Position)")
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(-1.2, 1.5)

plt.tight_layout()
plt.savefig("mpc_receding_horizon.png", dpi=300)
print("学术级滚动时域原理图生成完毕！请查看当前目录下的 mpc_receding_horizon.png")
