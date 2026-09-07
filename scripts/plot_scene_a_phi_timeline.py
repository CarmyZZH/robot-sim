"""
Generate Figure 6-3: Scene A configuration switching timeline.

The timeline is produced from ScriptedBrain's FSM thresholds and the same Phi
update rate used in evaluate_brain.py. It is a reproducible diagram for the
single narrow-gate scenario rather than a hand-drawn sketch.
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from brains.scripted_brain import ScriptedBrain


DT = 0.05
TOTAL_TIME = 14.0
OBSTACLE_TYPE = "narrow_gate"
INITIAL_PHI = 1.5
MAX_DELTA_PHI = 0.5 * DT


def configure_fonts():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def build_dummy_trajectory():
    x = np.linspace(0.0, 8.0, 300)
    y = np.zeros_like(x)
    yaw = np.zeros_like(x)
    return np.stack([x, y, yaw], axis=1)


def closest_distance_profile(t):
    """Synthetic Scene A distance-to-gate profile for FSM triggering."""
    if t < 1.0:
        return 2.2
    if t < 7.0:
        return 2.2 - (t - 1.0) * (3.2 / 6.0)
    return -1.0


def simulate_scene_a():
    brain = ScriptedBrain(dt=DT, mpc_horizon=10)
    global_traj = build_dummy_trajectory()

    current_phi = INITIAL_PHI
    records = []
    n_steps = int(TOTAL_TIME / DT) + 1

    for step in range(n_steps):
        t = step * DT
        closest_dist = closest_distance_profile(t)
        x = min(t * 0.35, 8.0)
        closest_type = OBSTACLE_TYPE if closest_dist > -0.8 else "none"
        obs = {
            "pose": (x, 0.0, 0.0),
            "closest_dist": closest_dist,
            "closest_type": closest_type,
        }

        _, target_phi = brain.get_action(obs, global_traj)
        if target_phi > current_phi:
            current_phi = min(target_phi, current_phi + MAX_DELTA_PHI)
        elif target_phi < current_phi:
            current_phi = max(target_phi, current_phi - MAX_DELTA_PHI)

        records.append(
            {
                "time": t,
                "state": brain.state,
                "closest_dist": closest_dist,
                "target_phi": target_phi,
                "current_phi": current_phi,
            }
        )

    return records


def state_segments(records):
    segments = []
    start = records[0]["time"]
    state = records[0]["state"]
    for previous, current in zip(records, records[1:]):
        if current["state"] != state:
            segments.append((start, previous["time"], state))
            start = current["time"]
            state = current["state"]
    segments.append((start, records[-1]["time"], state))
    return segments


def save_csv(records, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def draw_timeline(records, output_path):
    configure_fonts()
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_fonts()

    t = np.array([r["time"] for r in records])
    target_phi = np.array([r["target_phi"] for r in records])
    current_phi = np.array([r["current_phi"] for r in records])
    distance = np.array([r["closest_dist"] for r in records])

    fig, (ax_phi, ax_dist) = plt.subplots(
        2,
        1,
        figsize=(12.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    colors = {"CRUISE": "#E3F2FD", "PREPARE": "#FFF8E1", "MORPHING": "#FCE4EC"}
    labels = {"CRUISE": "CRUISE", "PREPARE": "PREPARE", "MORPHING": "MORPHING"}
    used_labels = set()
    for start, end, state in state_segments(records):
        label = labels[state] if state not in used_labels else None
        used_labels.add(state)
        ax_phi.axvspan(start, end, color=colors[state], alpha=0.85, label=label)
        ax_dist.axvspan(start, end, color=colors[state], alpha=0.85)

    ax_phi.plot(t, target_phi, color="#C62828", lw=2.6, ls="--", label="Target Phi")
    ax_phi.plot(t, current_phi, color="#1565C0", lw=2.8, label="Current Phi")
    ax_phi.set_ylabel("Topology Parameter Phi", fontsize=12)
    ax_phi.set_ylim(-0.08, 1.65)
    ax_phi.set_title("Scene A Configuration Switching Timeline", fontsize=17, fontweight="bold")
    ax_phi.legend(loc="upper right", ncol=5, fontsize=10, frameon=True)

    events = [1.0, 2.15, 2.55, 7.0]
    for event_t in events:
        ax_phi.axvline(event_t, color="#616161", lw=1.1, ls=":")

    ax_dist.plot(t, distance, color="#2E7D32", lw=2.2, label="Distance to Gate")
    ax_dist.axhline(1.6, color="#FF8F00", lw=1.6, ls="--", label="Detection Threshold 1.6 m")
    ax_dist.axhline(1.4, color="#D84315", lw=1.6, ls="--", label="Morph Trigger 1.4 m")
    ax_dist.axhline(-0.8, color="#6A1B9A", lw=1.4, ls="--", label="Clearance Threshold -0.8 m")
    ax_dist.set_xlabel("Time / s", fontsize=12)
    ax_dist.set_ylabel("Distance / m", fontsize=12)
    ax_dist.set_ylim(-1.15, 2.35)
    ax_dist.legend(loc="upper right", ncol=4, fontsize=9, frameon=True)

    fig.text(
        0.5,
        0.02,
        "Scene A: narrow-gate traversal. The FSM switches states by obstacle distance; target Phi changes from 1.5 to 0.0 and current Phi converges at 20 Hz.",
        ha="center",
        fontsize=10,
        color="#424242",
    )
    plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    records = simulate_scene_a()
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "scene_a_phi_timeline")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scene_a_phi_timeline.csv")
    image_path = os.path.join(output_dir, "scene_a_phi_timeline.png")
    save_csv(records, csv_path)
    draw_timeline(records, image_path)
    print(f"[OK] Timeline CSV saved to: {csv_path}")
    print(f"[OK] Timeline figure saved to: {image_path}")


if __name__ == "__main__":
    main()
