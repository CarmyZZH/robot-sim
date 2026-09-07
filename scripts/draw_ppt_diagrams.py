import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 全局样式
plt.style.use('ggplot')

# ==========================================================
# 方案 A: 热启动 (Warm-Start) 耗时对比图
# ==========================================================
def generate_scheme_A():
    fig, ax = plt.subplots(figsize=(6, 3))
    
    categories = ['Cold Start', 'Warm-Start']
    times = [18.5, 4.2] # 毫秒
    colors = ['#B0BEC5', '#E91E63'] # 灰白 和 亮红色
    
    # 画水平柱状图
    bars = ax.barh(categories, times, color=colors, height=0.5, edgecolor='black', linewidth=1)
    
    # 增加数值标注
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.1f} ms', ha='left', va='center', fontweight='bold', fontsize=12)
                
    # 在两条柱子之间画一个下降箭头和文字，表示性能提升
    ax.annotate(
        "Cpu Load Drop\n~77%", 
        xy=(10, 1.2), xycoords='data',
        xytext=(10, -0.2), textcoords='data',
        arrowprops=dict(arrowstyle="<-", color="green", lw=2, shrinkA=5, shrinkB=5),
        ha='center', va='center', color='green', fontweight='bold', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=1.5)
    )
    
    ax.set_xlabel('Solve Time per Frame (ms)', fontweight='bold')
    ax.set_title('Computational Efficiency: Warm-Start', fontweight='bold', fontsize=13)
    ax.set_xlim(0, 25)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "ppt_diagram_A_warm_start.png")
    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close()
    print(f"✅ 方案 A 图表已生成: {out_path}")

# ==========================================================
# 方案 B: 异步高低频交错控制流 (Control Loop Frequency)
# ==========================================================
def generate_scheme_B():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off') # 关闭坐标轴，纯画图
    
    # 定义画圆角矩形的便捷函数
    def draw_box(x, y, w, h, text, freq_text, color, ec='black'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                     edgecolor=ec, facecolor=color, lw=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + 0.2, text, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.4, freq_text, ha='center', va='center', fontsize=10, color='white', 
                bbox=dict(facecolor='black', alpha=0.2, edgecolor='none', pad=2))
                
    # 绘制三个层级的框
    # 1. 顶层大脑 (NavFormer)
    draw_box(2.5, 7.5, 5, 1.5, "Brain (Nav/DRL)", "Freq: 1 Hz (1000ms)", "#4CAF50")
    
    # 2. 中层小脑 (MPC)
    draw_box(1.5, 4.5, 7, 1.5, "Cerebellum (Kinematic MPC)", "Freq: 20 Hz (50ms)", "#E91E63")
    
    # 3. 底层真机 (Spinal & Motor)
    draw_box(0.5, 1.5, 9, 1.5, "Spinal Cord & Physics Engine", "Freq: 240 Hz (~4ms)", "#2196F3")
    
    # 画连接的箭头
    arrow_props = dict(facecolor='black', edgecolor='black', width=2, headwidth=8, headlength=8, shrink=0.0)
    
    # 箭头 1 -> 2
    ax.annotate('', xy=(5, 6.2), xytext=(5, 7.3), arrowprops=arrow_props)
    ax.text(5.2, 6.7, "Ref Traj (N=10)", va='center', fontsize=9, fontweight='bold', color='darkslategray')
    
    # 箭头 2 -> 3
    ax.annotate('', xy=(5, 3.2), xytext=(5, 4.3), arrowprops=arrow_props)
    ax.text(5.2, 3.7, "[Vx, Vy, ω] & Steering", va='center', fontsize=9, fontweight='bold', color='darkslategray')
    
    ax.set_title("Asynchronous Control Hierarchy", fontweight='bold', fontsize=14, y=0.95)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "ppt_diagram_B_frequency.png")
    plt.savefig(out_path, dpi=300, transparent=True)  # 透明背景方便贴入 PPT
    plt.close()
    print(f"✅ 方案 B 图表已生成: {out_path}")

if __name__ == '__main__':
    generate_scheme_A()
    generate_scheme_B()
