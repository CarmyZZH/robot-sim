# Phi-Conditioned Hierarchical Control for a Morphing Mobile Robot

This repository contains the research code for a PyBullet-based control system
for a continuously reconfigurable omnidirectional mobile robot. The central
idea is to use a shared, quantitative morphology context, called `Phi`, to
connect three control layers:

1. a brain layer for task and morphology planning;
2. a cerebellar layer for trajectory tracking and motion control;
3. a spinal layer for morphology decoding and constrained joint execution.

The current robot follows one predesigned continuous reconfiguration path, so
`Phi in [0, 3]` is a scalar coordinate on that one-dimensional morphology
manifold. It is not presented as a universal topology descriptor.

> **Source-only public release**
>
> The robot URDF, CAD meshes, raw datasets, trained weights, and experiment
> outputs are confidential and intentionally excluded from Git. A fresh clone
> is useful for reading and developing the control code, but it cannot reproduce
> the complete PyBullet experiments without the private local assets.

## Research Question

Can one observable morphology context coordinate task decisions, predictive
motion control, and joint-level execution more effectively than discrete mode
switching or independently tuned controllers?

The intended closed loop is:

```text
environment observation + task goal + estimated morphology
                            |
                            v
Brain: local path and target Phi
                            |
                            v
Cerebellum: trajectory MPC and morphology-aware swerve execution
                            |
                            v
Spinal layer: MorphGNN decoding, lock constraints, joint targets
                            |
                            v
                robot state and morphology feedback
```

The present code establishes the feed-forward execution path in simulation.
Explicitly conditioning the MPC model on measured morphology and closing the
morphology feedback loop on the physical robot are the next research steps.
The resulting system will test whether sharing the same morphology context
across all three layers improves tracking, transition stability, robustness,
and task success.

## Current Phi Representation

The implemented scalar `Phi` parameterizes four regions along a fixed
reconfiguration path:

| Phi range | Configuration region |
| --- | --- |
| `[0, 1)` | compact figure-eight-like configuration |
| `[1, 2]` | ring-like configuration |
| `(2, 3)` | gate-like to line-like transition |
| `3` | line-like configuration with locked joints |

Each morphology-graph node has ten input features:

```text
[7D node identity | normalized Phi | lock state | base angle]
```

For a broader family of robots, the scalar should become a learned or measured
morphology representation:

```text
z_phi = Encoder(graph, joint state, lock state, geometry, dynamics)
```

This generalization is a research objective, not a demonstrated result in the
current repository. The project uses the term *continuous reconfiguration*
unless module connectivity actually changes.

## System Architecture

### Brain Layer

`ScriptedBrain` is the current implemented upper-level baseline. It uses a
finite-state machine to select a local path segment and a target morphology.
The principal states are `CRUISE`, `PREPARE`, and `MORPHING`.

An experimental PPO pipeline is also included. It uses a 21-dimensional
observation and a three-dimensional action for planar motion and target
morphology. This branch is under development and should not yet be interpreted
as a successful learned planner.

### Cerebellar Layer

`KinematicMPC` tracks a ten-step planar reference trajectory at a nominal
`0.05 s` interval. It optimizes `[vx, vy, yaw_rate]` commands with SLSQP and
penalizes state error, control effort, and command variation.

Motion is handled in a world-referenced control pipeline because the geometric
wheel center moves relative to the base link during reconfiguration. Swerve
inverse kinematics and steering-reference calibration convert the optimized
motion into wheel commands. A morphology-aware PID gain schedule is available
for ablation experiments.

### Spinal Layer

`MorphGNN` maps the morphology graph and scalar `Phi` to seven morphing-joint
targets. It uses three GATv2 layers followed by an output MLP. Locked joints are
enforced with hard gating:

```text
q_target = q_raw * (1 - lock) + q_base * lock
```

A smaller two-layer `MorphGNNLite` model and analytical interpolation are
included as comparison methods.

## Benchmark Tasks

| Scene | Purpose |
| --- | --- |
| A | narrow-passage traversal with morphology switching |
| B | random multi-waypoint trajectory tracking |
| C | a compound course with several obstacle types |

The benchmark stack combines task generation, the scripted or learned brain,
MPC, swerve kinematics, morphology commands, and PyBullet execution.

## Current Evidence

These numbers are preliminary development results, not final paper claims:

- the retained MorphGNN training record reports a validation loss of
  approximately `0.00216` and a joint-angle MAE of approximately `0.026 rad`;
- three archived Scene B simulation runs reached the task criterion, with mean
  path errors in the approximate range of `0.019-0.030 m`;
- Phi-aware PID gain scheduling has shown only small and inconsistent gains;
- saved PPO policies have not yet demonstrated reliable benchmark completion;
- no physical-robot experiment has been completed.

Because datasets, checkpoints, and result artifacts are private, these values
are project status notes rather than a public reproducibility package.

## Repository Layout

```text
robot-sim/
|-- src/
|   |-- brains/          # upper-level interfaces and scripted policy
|   |-- controllers/     # MPC, PID, and swerve kinematics
|   |-- envs/            # PyBullet robot and benchmark environments
|   |-- models/          # MorphGNN models and graph datasets
|   `-- utils/           # morphology graph and coordinate utilities
|-- scripts/             # data, training, evaluation, and visualization tools
|-- tests/               # graph-contract and model-interface tests
|-- data/README.md       # private-data policy and expected local layout
|-- PAPER_DIRECTION.md   # paper thesis, hypotheses, and required baselines
|-- PAPER_READING_LIST.md # focused literature-reading plan
|-- total_plan.md        # implementation and experiment plan
`-- requirements.txt
```

## Private Local Prerequisites

Full simulation runs expect the confidential robot description to be restored
locally under:

```text
data/my_robot/
```

GNN evaluation and GUI inference additionally require a local checkpoint under
`experiments/`. Training and evaluation scripts write datasets and outputs to
ignored directories. Do not force-add these files to Git.

## Environment

Python 3.9 is the currently tested interpreter version. The project was
developed in a Conda environment with PyBullet, NumPy, SciPy, PyTorch, PyTorch
Geometric, Gym, Matplotlib, and Stable-Baselines3.

```bash
conda create -n robot-sim python=3.9 -y
conda activate robot-sim
pip install -r requirements.txt
```

For CUDA installations, install mutually compatible PyTorch and PyTorch
Geometric builds for the local driver and toolkit.

## Main Workflows

Generate the supervised morphology dataset:

```bash
python scripts/collect_gnn_data.py
```

Train the full or lightweight morphology decoder:

```bash
python scripts/train.py --model full
python scripts/train.py --model lite
```

Run the interactive PyBullet controller:

```bash
python scripts/run_mpc_gui.py
```

Run the three-layer scripted benchmark:

```bash
python scripts/evaluate_brain.py
```

Train or evaluate the experimental PPO brain:

```bash
python scripts/train_drl_brain.py
python scripts/evaluate_drl.py
```

## Evaluation and Ablation

```bash
python scripts/eval_gnn_ablation.py
python scripts/eval_gnn_attention.py
python scripts/eval_gnn_robustness.py
python scripts/eval_mpc_benchmark.py
python scripts/eval_pid_ablation.py
python scripts/eval_scene_b_tracking.py
python scripts/test_mpc_pure_kinematics.py
python scripts/test_mpc_pybullet.py
```

The paper-oriented comparison matrix should cover:

- discrete morphology ID versus scalar Phi versus a higher-dimensional
  morphology description;
- Phi in the spinal layer only versus the spinal and cerebellar layers versus
  all three layers;
- analytical mapping, lookup or spline interpolation, MLP, GCN, and GATv2;
- fixed control, per-configuration tuning, discrete gain scheduling, and
  continuous Phi-conditioned control;
- target Phi used as state versus measured or estimated actual morphology;
- nominal operation versus payload, friction, delay, slip, and joint-fault
  perturbations.

## Tests

The source-level tests do not require private robot geometry:

```bash
python -m unittest discover -s tests -v
```

They currently verify the seven-node morphology graph contract, graph-to-robot
joint ordering, full/lite model output interfaces, and notebook hygiene. The
notebook check prevents committed execution counts and embedded outputs.

## Research Roadmap

1. Estimate actual morphology from encoder and joint feedback instead of
   treating the target Phi command as the physical state.
2. Condition MPC geometry, constraints, and residual dynamics on morphology.
3. Complete the baseline and layer-wise ablation matrix with repeated trials
   and confidence intervals.
4. Add system identification and simulation randomization for mass, friction,
   delay, backlash, sensor bias, and wheel slip.
5. Deploy the same closed loop on the physical robot and report synchronized
   morphology, tracking, transition, energy, robustness, and success metrics.
6. Extend scalar Phi to `z_phi` only after collecting multiple morphology
   families, connection graphs, or structural-fault cases.

See [PAPER_DIRECTION.md](PAPER_DIRECTION.md) for the core thesis and
[PAPER_READING_LIST.md](PAPER_READING_LIST.md) for the curated reading plan.
