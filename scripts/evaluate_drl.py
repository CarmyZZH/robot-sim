import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pybullet as p

# 确保能找到项目根目录下的包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.envs.brain_gym_env import BrainGymEnv
from stable_baselines3 import PPO, SAC

def select_model(exp_base_dir):
    """交互式选择模型文件夹"""
    if not os.path.exists(exp_base_dir):
        return None, None
        
    subdirs = sorted([d for d in os.listdir(exp_base_dir) 
                      if d.startswith('drl_train') or d.startswith('drl_curriculum')], reverse=True)
    if not subdirs:
        return None, None
        
    print("\n[请选择要评估的训练记录]")
    for i, d in enumerate(subdirs):
        # 标注每个目录里有哪些模型文件
        d_path = os.path.join(exp_base_dir, d)
        models = [f for f in os.listdir(d_path) if f.endswith('.zip')]
        print(f"  [{i+1}] {d}  ({len(models)} 个模型: {', '.join(models[:3])}{'...' if len(models)>3 else ''})")
        
    try:
        choice = int(input(f"请输入编号 (1-{len(subdirs)}) [默认 1]: ").strip() or "1")
        selected_dir = os.path.join(exp_base_dir, subdirs[choice-1])
    except:
        selected_dir = os.path.join(exp_base_dir, subdirs[0])
        
    print(f"✓ 已选择: {os.path.basename(selected_dir)}")
    
    # 查找模型文件（按优先级）
    candidates = [
        "ppo_brain_final.zip",
        "best_model.zip",
        "ppo_phase3_task_only.zip",
        "ppo_phase2_mixed.zip",
        "ppo_phase1_imitation.zip",
        "sac_brain_final.zip",
    ]
    for c in candidates:
        path = os.path.join(selected_dir, c)
        if os.path.exists(path):
            algo = SAC if 'sac' in c else PPO
            print(f"  加载模型: {c}")
            return path, algo
    
    # fallback: 找子目录里的 best_model.zip
    for sub in os.listdir(selected_dir):
        sub_best = os.path.join(selected_dir, sub, "best_model.zip")
        if os.path.exists(sub_best):
            print(f"  加载模型: {sub}/best_model.zip")
            return sub_best, PPO
    
    return None, None

def run_single_eval(model, model_path, urdf_path, eval_dir):
    """模式 1: 单次 GUI 可视化评估"""
    env = BrainGymEnv(urdf_path, gui=True, scene_id='C', imitation_weight=0.0)
    obs, _ = env.reset()
    
    print("\n✅ GUI 环境已启动，开始推理展示...")
    done = False
    step_count = 0
    traj_x, traj_y, phi_records, reward_records = [], [], [], []
    
    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            traj_x.append(obs[0])
            traj_y.append(obs[1])
            phi_records.append(1.5 * action[2] + 1.5)
            reward_records.append(reward)
            step_count += 1
            if done or step_count > 100: break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    
    # 画图
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(traj_x, traj_y, '#E91E63', marker='o', markersize=3, label='Path')
    ax1.set_title("Robot Trajectory")
    ax1.axis('equal')
    ax2.plot(phi_records, '#2196F3', lw=2, label='Phi')
    ax2.set_title("Morphing History")
    plt.savefig(os.path.join(eval_dir, 'single_eval_plot.png'), dpi=200)
    print(f"✓ 轨迹图已保存至: {eval_dir}")

def run_batch_eval(model, model_path, urdf_path, eval_dir, num_episodes=10):
    """模式 2: 批量非 GUI 统计评估"""
    print(f"\n📊 开始批量统计评估 (共 {num_episodes} 轮)...")
    env = BrainGymEnv(urdf_path, gui=False, scene_id='C', imitation_weight=0.0)
    
    all_rewards = []
    success_count = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if reward > 50: # 命中终点大奖
                success_count += 1
                
        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: Reward = {ep_reward:.2f}, {'SUCCESS' if reward > 50 else 'FAIL'}")
        
    env.close()
    
    # 统计分析与画图
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_episodes+1), all_rewards, color='teal', alpha=0.7)
    plt.axhline(y=np.mean(all_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(all_rewards):.2f}')
    plt.title(f"Batch Evaluation Results (Success Rate: {success_count/num_episodes*100:.1f}%)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(os.path.join(eval_dir, 'batch_stats.png'), dpi=200)
    
    # 保存原始数据
    np.save(os.path.join(eval_dir, 'rewards.npy'), np.array(all_rewards))
    print(f"\n✅ 批量评估完成！成功率: {success_count/num_episodes*100:.1f}%")
    print(f"统计图表已保存至: {eval_dir}")

def main():
    print("==================================================")
    print(" 🚀 DRL 大脑全链路评估工具 (多模式版) ")
    print("==================================================")
    
    exp_base_dir = os.path.join(project_root, "experiments")
    model_path, algo_class = select_model(exp_base_dir)
    
    if not model_path:
        print("❌ 找不到有效的模型，退出。")
        return
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = algo_class.load(model_path, device=device)
    
    print("\n[请选择评估模式]")
    print("  1. 单次演示模式 (带 GUI，生成轨迹图)")
    print("  2. 批量统计模式 (无 GUI，跑 10 轮生成性能报表)")
    mode = input("请输入编号 (1/2) [默认 1]: ").strip() or "1"
    
    # 建立结果文件夹
    ts = time.strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(exp_base_dir, f"eval_report_{ts}")
    os.makedirs(eval_dir, exist_ok=True)
    
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    
    if mode == "1":
        run_single_eval(model, model_path, urdf_path, eval_dir)
    else:
        run_batch_eval(model, model_path, urdf_path, eval_dir)

if __name__ == "__main__":
    main()
