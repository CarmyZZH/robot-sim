"""
Evaluate Scene B multi-waypoint cruising without GUI or side air walls.

The script keeps Scene B's original random waypoint generation, runs the same
ScriptedBrain -> MPC -> Swerve -> PyBullet control chain, records the robot
centroid trajectory, and saves paper-ready trajectory/summary figures.
"""

import argparse
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from brains.scripted_brain import ScriptedBrain
from controllers.mpc_controller import KinematicMPC
from controllers.swerve_kinematics import SwerveDriveController
from envs.benchmark_env import BenchmarkEnv
from scripts.test_mpc_vs_pid_topology import ZeroYawSmoother, compute_angles_from_phi


DT = 0.05


def configure_fonts():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def nearest_path_errors(points, reference):
    ref_xy = reference[:, :2]
    errors = []
    for point in points:
        d = np.linalg.norm(ref_xy - point[:2], axis=1)
        errors.append(float(np.min(d)))
    return np.asarray(errors)


def run_trial(seed, max_time):
    np.random.seed(seed)
    urdf_path = os.path.join(PROJECT_ROOT, "data", "my_robot", "robot1231_2.urdf")
    env = BenchmarkEnv(urdf_path, gui=False, scene_id="B", add_boundaries=False)
    brain = ScriptedBrain(dt=DT, mpc_horizon=10)
    mpc = KinematicMPC(dt=DT, N=10)
    swerve = SwerveDriveController(wheel_radius=env.core_env.WHEEL_RADIUS)
    smoother = ZeroYawSmoother(alpha=0.9)

    try:
        obs = env.reset()
        global_traj = env.task.global_traj
        block_angles = compute_angles_from_phi(1.5)

        for _ in range(20):
            env.core_env.step({"blocks": block_angles, "velocity": [0] * 4, "steering": [0] * 4})

        last_cmd = np.zeros(3)
        current_phi = 1.5
        records = []
        success = False
        info = {"dist_to_goal": float("inf"), "success": False}
        max_steps = int(max_time / DT)

        for step in range(max_steps):
            obs = env.get_obs()
            ref_traj_10, target_phi = brain.get_action(obs, global_traj)
            target_phi = 1.5

            x_w, y_w, yaw_w = obs["pose"]
            current_state = np.array([x_w, y_w, yaw_w])
            cmd = mpc.solve(current_state, ref_traj_10, last_cmd)
            last_cmd = cmd

            raw_zero_yaws = env.core_env.get_steering_zero_yaws_imu()
            zero_yaws_w = smoother.update(raw_zero_yaws)
            for i, s_id in enumerate(env.core_env.steering_ids):
                corner = env.core_env.STEER_CORNER_ORDER[i]
                swerve.last_angles[corner] = p.getJointState(env.core_env.robotId, s_id)[0]

            vx_b, vy_b, omega = cmd
            vx_world = vx_b * np.cos(yaw_w) - vy_b * np.sin(yaw_w)
            vy_world = vx_b * np.sin(yaw_w) + vy_b * np.cos(yaw_w)
            cmds_sw = swerve.compute_swerve_kinematics(
                vx_world,
                vy_world,
                omega,
                env.core_env.get_wheel_positions_for_kinematics()[0],
                zero_yaws_w,
            )

            steers = [cmds_sw[env.core_env.STEER_CORNER_ORDER[i]]["angle"] for i in range(4)]
            vels = [cmds_sw[env.core_env.STEER_CORNER_ORDER[i]]["speed"] for i in range(4)]

            max_delta_phi = 0.5 * DT
            if target_phi > current_phi:
                current_phi = min(target_phi, current_phi + max_delta_phi)
            elif target_phi < current_phi:
                current_phi = max(target_phi, current_phi - max_delta_phi)

            val_action = {"steering": steers, "velocity": vels, "blocks": compute_angles_from_phi(current_phi)}
            obs, success, info = env.step(val_action)
            records.append([step * DT, obs["pose"][0], obs["pose"][1], obs["pose"][2], info["dist_to_goal"]])
            if success:
                break

        points = np.asarray(records)
        errors = nearest_path_errors(points[:, 1:3], global_traj) if len(points) else np.asarray([np.nan])
        metrics = {
            "seed": seed,
            "success": bool(success),
            "duration_s": float(points[-1, 0]) if len(points) else 0.0,
            "mean_path_error_m": float(np.nanmean(errors)),
            "max_path_error_m": float(np.nanmax(errors)),
            "final_goal_dist_m": float(info["dist_to_goal"]),
            "num_waypoints": int(len(env.task.waypoints)),
        }
        return {
            "records": points,
            "errors": errors,
            "reference": global_traj,
            "waypoints": env.task.waypoints.copy(),
            "goal_pose": env.task.goal_pose,
            "metrics": metrics,
        }
    finally:
        env.core_env.close()


def draw_representative(result, output_path):
    configure_fonts()
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_fonts()

    records = result["records"]
    reference = result["reference"]
    waypoints = result["waypoints"]
    metrics = result["metrics"]

    fig, (ax_path, ax_err) = plt.subplots(1, 2, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax_path.plot(reference[:, 0], reference[:, 1], color="#111111", lw=2.4, ls="--", label="Reference Spline")
    ax_path.plot(records[:, 1], records[:, 2], color="#1565C0", lw=2.6, label="Robot Centroid")
    ax_path.scatter(waypoints[:, 0], waypoints[:, 1], s=70, color="#2E7D32", zorder=5, label="Random Waypoints")
    ax_path.scatter([reference[0, 0]], [reference[0, 1]], s=90, marker="o", color="#43A047", label="Start")
    ax_path.scatter([reference[-1, 0]], [reference[-1, 1]], s=110, marker="*", color="#C62828", label="Goal")
    ax_path.set_title("Scene B Multi-waypoint Cruising", fontsize=16, fontweight="bold")
    ax_path.set_xlabel("x / m")
    ax_path.set_ylabel("y / m")
    ax_path.axis("equal")
    ax_path.legend(loc="best", fontsize=10)

    ax_err.plot(records[:, 0], result["errors"], color="#EF6C00", lw=2.2)
    ax_err.set_title("Path Error over Time", fontsize=15, fontweight="bold")
    ax_err.set_xlabel("Time / s")
    ax_err.set_ylabel("Nearest Path Error / m")
    ax_err.text(
        0.04,
        0.96,
        f"Mean Error: {metrics['mean_path_error_m']:.3f} m\nMax Error: {metrics['max_path_error_m']:.3f} m\nFinal Distance: {metrics['final_goal_dist_m']:.3f} m",
        transform=ax_err.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BDBDBD"},
        fontsize=11,
    )

    fig.text(0.5, 0.02, "Headless Scene B run: random waypoints and spline reference are preserved; side air walls are disabled; robot centroid trajectory is recorded.", ha="center", fontsize=10)
    plt.tight_layout(rect=[0.03, 0.05, 0.98, 0.97])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_summary(results, output_path):
    configure_fonts()
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_fonts()

    mean_errors = np.array([r["metrics"]["mean_path_error_m"] for r in results])
    max_errors = np.array([r["metrics"]["max_path_error_m"] for r in results])
    final_dists = np.array([r["metrics"]["final_goal_dist_m"] for r in results])
    success_rate = np.mean([r["metrics"]["success"] for r in results]) * 100.0

    labels = ["Mean Path Error", "Max Path Error", "Final Distance"]
    means = [mean_errors.mean(), max_errors.mean(), final_dists.mean()]
    stds = [mean_errors.std(), max_errors.std(), final_dists.std()]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.bar(labels, means, yerr=stds, color=["#1565C0", "#EF6C00", "#2E7D32"], alpha=0.88, capsize=8)
    ax.set_title("Scene B Multi-trial Cruising Statistics", fontsize=17, fontweight="bold")
    ax.set_ylabel("Distance / m")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=12)
    ax.text(0.98, 0.95, f"Trials: {len(results)}\nSuccess Rate: {success_rate:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metrics(results, output_path):
    fields = list(results[0]["metrics"].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result["metrics"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scene B tracking without GUI or side walls.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "experiments", f"scene_b_tracking_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i in range(args.trials):
        seed = args.seed + i
        print(f"[Trial {i + 1}/{args.trials}] seed={seed}")
        result = run_trial(seed, args.max_time)
        results.append(result)
        print(
            f"  mean={result['metrics']['mean_path_error_m']:.3f}m, "
            f"max={result['metrics']['max_path_error_m']:.3f}m, "
            f"final={result['metrics']['final_goal_dist_m']:.3f}m, "
            f"success={result['metrics']['success']}"
        )

    representative = min(results, key=lambda r: r["metrics"]["mean_path_error_m"])
    draw_representative(representative, os.path.join(output_dir, "scene_b_tracking_trajectory.png"))
    draw_summary(results, os.path.join(output_dir, "scene_b_tracking_summary.png"))
    save_metrics(results, os.path.join(output_dir, "scene_b_tracking_metrics.csv"))

    print(f"[OK] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
