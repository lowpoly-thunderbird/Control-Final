import numpy as np
from simulator import Simulator_new as Simulator
from pathlib import Path
import os
from typing import Dict
import pinocchio as pin
import matplotlib.pyplot as plt

# Load the robot model from scene XML
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

def plot_results(times: np.ndarray, positions:np.ndarray, pos_err: np.ndarray, control: np.ndarray):
    """Plot and save simulation results."""

    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/SID_positions.png')
    plt.close()
    
    # Joint positions errors plot
    plt.figure(figsize=(10, 6))
    for i in range(pos_err.shape[1]):
        plt.plot(times, pos_err[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Position Errors [rad]')
    plt.title('Joint Position Errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/SID_position_errors.png')
    plt.close()

    # Joint controls plot
    plt.figure(figsize=(10, 6))
    for i in range(control.shape[1]):
        plt.plot(times, control[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint control signals')
    plt.title('Joint control signals over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/SID_signals.png')
    plt.close()

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """
    # Control gains tuned for UR5e
    kp = np.diag([100, 100, 100, 100, 100, 100])
    kd = np.diag([20, 20, 20, 20, 20, 20])
    
    # Target joint configuration
    q_des = np.array([-1.4, -1.3, 1., 0, 0, 1.57])
    dq_des = np.zeros(6)
    ddq_des = np.zeros(6)

    # Compute dynamics
    pin.computeAllTerms(model, data, q, dq)
    M_hat = 1.2 * data.M
    # Mass matrix
    h_hat = 1.2 * data.nle

    #Errors
    q_err = q_des - q
    print(q_err)

    dq_err = dq_des - dq
    
    # Control law
    tau = M_hat @ (ddq_des + kd @ dq_err + kp @ q_err) + h_hat
    return tau, q_err

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,  # Using joint space control
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/SID.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    
    # Set joint damping (example values, adjust as needed)
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)
    
    # Set joint friction (example values, adjust as needed)
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)
    
    # Get original properties
    ee_name = "end_effector"
    
    original_props = sim.get_body_properties(ee_name)
    print(f"\nOriginal end-effector properties:")
    print(f"Mass: {original_props['mass']:.3f} kg")
    print(f"Inertia:\n{original_props['inertia']}")
    
    # Add the end-effector mass and inertia
    sim.modify_body_properties(ee_name, mass=3)
    # Print modified properties
    props = sim.get_body_properties(ee_name)
    print(f"\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")
    
    # Set controller and run simulation
    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

    plot_results(sim.times, sim.positions, sim.pos_err, sim.controls)

if __name__ == "__main__":
    main() 