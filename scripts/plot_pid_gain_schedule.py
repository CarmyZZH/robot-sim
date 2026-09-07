"""
Draw the Phi-aware PID gain scheduling curve for thesis Figure 5-5.

The plotted values are taken from VelocityPIDController itself, so the figure
stays consistent with the controller implementation.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from controllers.pid_controller import VelocityPIDController


def collect_gain_schedule(phi_values):
    controller = VelocityPIDController()
    schedule = {
        "phi": [],
        "vx_kp": [],
        "vx_ki": [],
        "vx_kd": [],
        "omega_kp": [],
        "omega_ki": [],
        "omega_kd": [],
        "scale": [],
    }

    base_vx = VelocityPIDController.DEFAULT_GAINS["vx"]

    for phi in phi_values:
        controller.update_gains_for_phi(float(phi))
        scale = controller.pid_vx.kp / base_vx["kp"]

        schedule["phi"].append(phi)
        schedule["vx_kp"].append(controller.pid_vx.kp)
        schedule["vx_ki"].append(controller.pid_vx.ki)
        schedule["vx_kd"].append(controller.pid_vx.kd)
        schedule["omega_kp"].append(controller.pid_omega.kp)
        schedule["omega_ki"].append(controller.pid_omega.ki)
        schedule["omega_kd"].append(controller.pid_omega.kd)
        schedule["scale"].append(scale)

    return {key: np.asarray(value) for key, value in schedule.items()}


def configure_fonts():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def draw_schedule(schedule, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_fonts()

    fig, (ax_gain, ax_scale) = plt.subplots(
        1,
        2,
        figsize=(12.8, 7.2),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    phi = schedule["phi"]

    ax_gain.plot(phi, schedule["vx_kp"], color="#1565C0", lw=2.6, label="vx/vy Kp")
    ax_gain.plot(phi, schedule["vx_ki"], color="#42A5F5", lw=2.4, label="vx/vy Ki")
    ax_gain.plot(phi, schedule["omega_kp"], color="#C62828", lw=2.6, label="omega Kp")
    ax_gain.plot(phi, schedule["omega_ki"], color="#EF5350", lw=2.4, label="omega Ki")
    ax_gain.plot(phi, schedule["vx_kd"], color="#757575", lw=1.8, ls="--", label="vx/vy Kd (fixed)")
    ax_gain.plot(phi, schedule["omega_kd"], color="#9E9E9E", lw=1.8, ls="--", label="omega Kd (fixed)")

    for marker_phi, label in [(0.0, "Compact / 8-shape"), (1.5, "O-shape baseline"), (3.0, "Line shape")]:
        ax_gain.axvline(marker_phi, color="#BDBDBD", lw=1.2, ls=":")
        ax_gain.text(
            marker_phi,
            ax_gain.get_ylim()[1] * 0.96,
            label,
            ha="center",
            va="top",
            fontsize=10,
            color="#424242",
        )

    ax_gain.set_title("Phi-aware PID Gain Schedule", fontsize=17, fontweight="bold")
    ax_gain.set_xlabel("Topology Parameter Phi", fontsize=13)
    ax_gain.set_ylabel("PID Gain Value", fontsize=13)
    ax_gain.set_xlim(0, 3)
    ax_gain.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=True)

    ax_scale.plot(phi, schedule["scale"], color="#2E7D32", lw=3.0)
    ax_scale.fill_between(phi, 1.0, schedule["scale"], color="#A5D6A7", alpha=0.45)
    ax_scale.set_title("Gain Scale", fontsize=15, fontweight="bold")
    ax_scale.set_xlabel("Topology Parameter Phi", fontsize=13)
    ax_scale.set_ylabel("Scale", fontsize=13)
    ax_scale.set_xlim(0, 3)
    ax_scale.set_ylim(0.98, 1.22)
    ax_scale.set_yticks([1.0, 1.1, 1.2])
    ax_scale.annotate(
        "Minimum 1.0x\nPhi=1.5",
        xy=(1.5, 1.0),
        xytext=(1.05, 1.06),
        arrowprops={"arrowstyle": "->", "color": "#2E7D32", "lw": 1.4},
        fontsize=10,
        color="#2E7D32",
    )
    ax_scale.annotate(
        "Maximum 1.2x",
        xy=(3.0, 1.2),
        xytext=(2.05, 1.17),
        arrowprops={"arrowstyle": "->", "color": "#2E7D32", "lw": 1.4},
        fontsize=10,
        color="#2E7D32",
    )

    fig.suptitle("PID Gain Scheduling Curve with Phi", fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Kp and Ki are continuously scheduled by Phi; Kd remains fixed. Values are generated from the implemented controller.",
        ha="center",
        fontsize=11,
        color="#424242",
    )

    plt.tight_layout(rect=[0.03, 0.07, 0.98, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    phi_values = np.linspace(0.0, 3.0, 301)
    schedule = collect_gain_schedule(phi_values)
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "pid_gain_schedule")
    output_path = os.path.join(output_dir, "pid_gain_schedule_curve.png")
    draw_schedule(schedule, output_path)
    print(f"[OK] PID gain schedule curve saved to: {output_path}")


if __name__ == "__main__":
    main()
