import os
import sys
import time
import numpy as np
import torch
import pybullet as p
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from envs.robot_env import MorphingRobotEnv
from models.model import MorphGNN
from utils.graph_converter import RobotGraphConverter

def formula_angles(phi: float) -> list:
    phi = np.clip(phi, 0.0, 3.0)
    if phi <= 1.0:
        r = phi
        ca = 1.57 * (1.0 - r)
        return [ca, 1.57, ca, -1.57, ca, 1.57, ca]
    elif phi <= 2.0:
        r = 2.0 - phi
        j4 = -1.57 * (2.0 * r - 1.0)
        return [0.0, -j4, 0.0, j4, 0.0, -j4, 0.0]
    return [0]*7

def get_physical_shape_error(env, ideal_angles):
    current_angles = []
    for b_id in env.block_ids:
        current_angles.append(p.getJointState(env.robotId, b_id)[0])
    err = np.mean((np.array(current_angles) - np.array(ideal_angles))**2)
    return err

def run_simulation_test(env, model, converter, device, use_gnn=True, fail_idx=3):
    env.reset()
    phis = np.linspace(1.5, 0.0, 100)
    failure_step = 30
    edge_index = converter.edge_index_7.to(device)
    
    errors = []
    frozen_angle = 0.0
    is_failed = False
    
    for step, phi in enumerate(phis):
        ideal = formula_angles(phi)
        
        if use_gnn:
            features = converter.get_feature_matrix(phi)
            if is_failed:
                features[fail_idx, 8] = 1.0 
                features[fail_idx, 9] = frozen_angle
            
            x = torch.tensor(features, dtype=torch.float32).to(device)
            with torch.no_grad():
                # 显式使用 predict 接口确保处于 eval 模式
                pred = model.predict(x, edge_index).squeeze().cpu().numpy()
            cmds = converter.graph_to_env_angles(pred)
        else:
            cmds = formula_angles(phi)
            
        action = {
            'blocks': cmds,
            'velocity': [0]*4,
            'steering': [0]*4
        }
        
        if step == failure_step:
            is_failed = True
            frozen_angle = p.getJointState(env.robotId, env.block_ids[fail_idx])[0]
            
        if is_failed:
            p.setJointMotorControl2(env.robotId, env.block_ids[fail_idx], 
                                    p.POSITION_CONTROL, targetPosition=frozen_angle, force=100)
            
        env.step(action)
        errors.append(get_physical_shape_error(env, ideal))
        
    return np.array(errors)

def main():
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    env = MorphingRobotEnv(urdf_path, gui=False, control_steps=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    converter = RobotGraphConverter()
    
    exp_dir = sorted([d for d in os.listdir(os.path.join(project_root, "experiments")) if d.startswith('phi_run_')], reverse=True)[0]
    model_path = os.path.join(project_root, "experiments", exp_dir, "best_model.pt")
    model = MorphGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval() # ★ 核心修复：进入评估模式
    
    fail_idx = 3 
    
    print(">> 运行公式法失效仿真...")
    fml_errors = run_simulation_test(env, model, converter, device, use_gnn=False, fail_idx=fail_idx)
    
    print(">> 运行 GNN 法失效仿真...")
    gnn_errors = run_simulation_test(env, model, converter, device, use_gnn=True, fail_idx=fail_idx)
    
    env.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(gnn_errors, label='GNN (Adaptive Spinal Cord)', color='green', lw=2)
    plt.plot(fml_errors, label='Formula (Brute-force)', color='red', lw=2, linestyle='--')
    plt.axvline(x=30, color='orange', linestyle=':', label='Joint Failure')
    plt.yscale('log')
    plt.title('Physical Shape Distortion after Joint Failure')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Squared Error of All Joints')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(project_root, 'experiments', f'gnn_robustness_sim_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'sim_robustness_comparison.png'), dpi=200)
    
    print(f"✅ 仿真实验完成。结果保存至: {out_dir}")

if __name__ == "__main__":
    main()
