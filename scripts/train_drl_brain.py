"""
DRL 大脑分阶段课程训练脚本

训练策略：
  Phase 1 (imitation)  — 高模仿权重，PPO 快速学会障碍物-形态关联
  Phase 2 (mixed)      — 降低模仿权重，让策略开始依赖任务奖励
  Phase 3 (task_only)  — 关闭模仿，纯任务奖励微调

每个 Phase 结束后自动保存模型，支持断点续传。
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.envs.brain_gym_env import BrainGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common import results_plotter


# ── 课程配置 ──────────────────────────────────────────────
CURRICULUM = [
    {
        "name": "phase1_imitation",
        "scene_id": "C",
        "steps": 15_000,
        "imitation_weight": 1.0,
        "desc": "高模仿权重，学习障碍物-形态关联",
    },
    {
        "name": "phase2_mixed",
        "scene_id": "C",
        "steps": 15_000,
        "imitation_weight": 0.3,
        "desc": "降低模仿，开始依赖任务奖励",
    },
    {
        "name": "phase3_task_only",
        "scene_id": "C",
        "steps": 20_000,
        "imitation_weight": 0.0,
        "desc": "关闭模仿，纯任务奖励微调",
    },
]


def make_env(urdf_path, log_dir, scene_id='C', imitation_weight=1.0,
             max_steps=60, expert_lookahead_idx=5):
    """工厂函数，返回可被 DummyVecEnv 调用的 callable。"""
    def _init():
        env = BrainGymEnv(
            urdf_path,
            gui=False,
            scene_id=scene_id,
            max_steps=max_steps,
            imitation_weight=imitation_weight,
            expert_lookahead_idx=expert_lookahead_idx,
        )
        env = Monitor(env, log_dir)
        return env
    return _init


def find_latest_checkpoint(exp_root):
    """在 experiments/ 下找最近的 drl_curriculum_* 目录里的模型文件。"""
    if not os.path.exists(exp_root):
        return None, None
    dirs = sorted([d for d in os.listdir(exp_root)
                   if d.startswith('drl_curriculum_')], reverse=True)
    for d in dirs:
        d_path = os.path.join(exp_root, d)
        # 按 phase 倒序找
        for phase in reversed(CURRICULUM):
            candidate = os.path.join(d_path, f"ppo_{phase['name']}.zip")
            if os.path.exists(candidate):
                return candidate, phase['name']
        # fallback
        for f in ['best_model.zip', 'ppo_brain_final.zip']:
            if os.path.exists(os.path.join(d_path, f)):
                return os.path.join(d_path, f), None
    return None, None


def main():
    parser = argparse.ArgumentParser(description='DRL Brain Curriculum Training')
    parser.add_argument('--resume', action='store_true', help='从最近存档续传')
    parser.add_argument('--start-phase', type=int, default=None,
                        help='直接从指定阶段开始 (1/2/3)，跳过之前的阶段')
    parser.add_argument('--steps-multiplier', type=float, default=1.0,
                        help='步数倍率，默认 1.0')
    parser.add_argument('--gui', action='store_true', help='训练时显示 GUI（仅调试）')
    args = parser.parse_args()

    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("  DRL 大脑分阶段课程训练")
    print(f"  设备: {device.upper()}")
    print("=" * 60)

    # ── 交互式菜单（无命令行参数时生效）──
    if args.start_phase is None and not args.resume:
        print("\n[Training Mode]")
        print("  1. [New] Start fresh (Phase 1 -> 2 -> 3)")
        print("  2. [Resume] Continue from checkpoint")
        print("  3. [Skip] Start from Phase 2 (load Phase 1 model)")
        print("  4. [Skip] Start from Phase 3 (load latest model)")
        try:
            choice = int(input("\n请输入 (1/2/3/4) [默认 1]: ").strip() or "1")
        except (ValueError, EOFError):
            choice = 1

        if choice == 2:
            args.resume = True
        elif choice == 3:
            args.start_phase = 2
        elif choice == 4:
            args.start_phase = 3
        # choice == 1: 保持默认，从头开始

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(project_root, 'experiments', f'drl_curriculum_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    print(f"  输出目录: {exp_dir}")

    # ── 断点续传 / 指定阶段 ──
    model = None
    start_phase_idx = 0

    # --start-phase 优先级高于 --resume
    if args.start_phase is not None:
        start_phase_idx = max(0, args.start_phase - 1)  # 用户输入1/2/3，转为0-indexed
        print(f"\n⏩ 直接从 Phase {args.start_phase} 开始")
        # 尝试加载前一阶段的模型作为起点
        if start_phase_idx > 0:
            prev_phase_name = CURRICULUM[start_phase_idx - 1]['name']
            ckpt_path, _ = find_latest_checkpoint(
                os.path.join(project_root, 'experiments'))
            if ckpt_path:
                print(f"   加载前序模型: {ckpt_path}")
                tmp_env = DummyVecEnv([make_env(urdf_path, exp_dir)])
                tmp_env = VecNormalize(tmp_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
                model = PPO.load(ckpt_path, env=tmp_env, device=device)
            else:
                print("   [WARN] No previous model found, starting from scratch.")
    elif args.resume:
        ckpt_path, last_phase = find_latest_checkpoint(
            os.path.join(project_root, 'experiments'))
        if ckpt_path:
            print(f"\n[Resume] Loading checkpoint: {ckpt_path}")
            # 找到该 phase 之后的索引
            if last_phase:
                for i, ph in enumerate(CURRICULUM):
                    if ph['name'] == last_phase:
                        start_phase_idx = i + 1
                        break
            if start_phase_idx >= len(CURRICULUM):
                print("[DONE] All phases completed, no further training needed.")
                return
            tmp_env = DummyVecEnv([make_env(urdf_path, exp_dir)])
            tmp_env = VecNormalize(tmp_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO.load(ckpt_path, env=tmp_env, device=device)
            print(f"   将从 Phase {start_phase_idx + 1}/{len(CURRICULUM)} 开始")
        else:
            print("[WARN] No checkpoint found, starting from scratch.")

    # ── 分阶段训练 ──
    for phase_idx in range(start_phase_idx, len(CURRICULUM)):
        phase = CURRICULUM[phase_idx]
        phase_steps = int(phase['steps'] * args.steps_multiplier)
        phase_dir = os.path.join(exp_dir, phase['name'])
        os.makedirs(phase_dir, exist_ok=True)

        print(f"\n{'─' * 50}")
        print(f"  Phase {phase_idx + 1}/{len(CURRICULUM)}: {phase['name']}")
        print(f"  {phase['desc']}")
        print(f"  场景: {phase['scene_id']}  |  模仿权重: {phase['imitation_weight']}")
        print(f"  训练步数: {phase_steps}")
        print(f"{'─' * 50}")

        # 创建训练环境
        train_env = DummyVecEnv([make_env(
            urdf_path, phase_dir,
            scene_id=phase['scene_id'],
            imitation_weight=phase['imitation_weight'],
        )])
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # 创建评估环境（始终关闭模仿，测真实任务表现）
        eval_dir = os.path.join(phase_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        eval_env = DummyVecEnv([make_env(
            urdf_path, eval_dir,
            scene_id=phase['scene_id'],
            imitation_weight=0.0,
        )])
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False,
            training=False, clip_obs=10.0)

        # 初始化或更新模型
        if model is None:
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                device=device,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=128,
                n_epochs=10,
                gamma=0.98,
                gae_lambda=0.95,
                ent_coef=0.02,
                clip_range=0.15,
            )
        else:
            model.set_env(train_env)

        # 回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=phase_dir,
            log_path=phase_dir,
            eval_freq=max(phase_steps // 10, 500),
            deterministic=True,
            render=False,
        )
        ckpt_callback = CheckpointCallback(
            save_freq=max(phase_steps // 5, 1000),
            save_path=phase_dir,
            name_prefix=f"ckpt_{phase['name']}",
        )

        # 训练
        model.learn(
            total_timesteps=phase_steps,
            callback=[eval_callback, ckpt_callback],
            reset_num_timesteps=False,
        )

        # 保存 phase 模型
        model.save(os.path.join(exp_dir, f"ppo_{phase['name']}"))
        train_env.save(os.path.join(exp_dir, f"vecnorm_{phase['name']}.pkl"))
        print(f"[DONE] Phase {phase_idx + 1} completed, model saved.")

        train_env.close()
        eval_env.close()

    # ── 最终保存 ──
    final_path = os.path.join(exp_dir, "ppo_brain_final")
    model.save(final_path)
    print(f"\n[DONE] Training complete! Final model: {final_path}.zip")

    # 尝试画学习曲线
    try:
        results_plotter.plot_results(
            [exp_dir], sum(p['steps'] for p in CURRICULUM),
            results_plotter.X_TIMESTEPS, "PPO Curriculum Learning Curve")
        plt.savefig(os.path.join(exp_dir, 'learning_curve.png'),
                    dpi=200, bbox_inches='tight')
        print(f"[DONE] Learning curve: {exp_dir}/learning_curve.png")
    except Exception as e:
        print(f"[WARN] Failed to plot learning curve: {e}")


if __name__ == "__main__":
    main()
