"""Joint space control example for robot manipulation.

This example demonstrates basic joint space control of a UR5e robot arm using
PD (Proportional-Derivative) control. The robot moves from its initial configuration
to a target joint configuration while minimizing position and velocity errors.

Key Concepts Demonstrated:
    - Joint space PD control implementation
    - Real-time simulation visualization
    - Video recording of simulation
    - Basic robot state access and control

Example:
    To run this example:
    
    $ python 01_joint_space.py

Notes:
    - The controller gains (kp, kd) are tuned for the UR5e robot
    - The target configuration (q0) is set to a predefined pose
    - The simulation runs for 10 seconds with real-time visualization
"""

import numpy as np
from simulator import Simulator
from pathlib import Path
import numpy as np
import pinocchio as pin
import os
import mujoco

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float, sim=None) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """
  
    # Control gains tuned for UR5e
    kp = np.array([100, 100, 100, 100, 100, 100])
    kd = np.array([20, 20, 20, 20, 20, 20])
    
    # Target joint configuration
    q0 = np.array([-1.4, -1.3, 1, 0, 0, 0])
    dq_d = np.zeros(6)    # Скорость цели = 0
    ddq_d = np.zeros(6)   # Ускорение цели = 0
    
    # Load the robot model from scene XML
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()

    # Compute all dynamics quantities at once
    pin.computeAllTerms(model, data, q, dq)

    # Mass matrix
    M = data.M
    # Nonlinear effects (Coriolis + gravity)
    nle = data.nle
    print(q)

    # Матрица Lambda (положительно определенная)
    Lambda = np.diag([15 for i in range(6)])
    # Lambda = np.diag([2.0, 2.0, 1.5, 1.0, 1.0, 0.8])  # Гораздо меньше!

    # Ошибки
    e = q0 - q  # Ошибка положения
    de = dq_d - dq  # Ошибка скорости
    
    # Скользящая поверхность: s = de + Λ·e
    s = de + Lambda @ e
    
    # Норма скользящей поверхности
    s_norm = np.linalg.norm(s)
    
    # 1 вариант как в лекции - оказался провальным:
    # rho = (k / sigma_max) * M_inv
    # v_s = rho @ s / s_norm

    # 2 вариант
    K_robust = 80.0
    v_s = (K_robust / s_norm) * s
    
    # Вспомогательный сигнал v
    v = ddq_d + Lambda @ de + v_s
    
    # Основное управление: u = M·v + С + g
    tau = M @ v + nle
    
    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning real-time joint space control...")
    sim = Simulator(
        # xml_path="scene.xml",
        xml_path="./robots/universal_robots_ur5e/scene2.xml",
        record_video=True,
        video_path="logs/videos/3_SMC.mp4",
        width=1920,
        height=1080
    )
    # Set joint damping coefficients
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)

    # Set joint friction coefficients
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)

    # Modify end-effector mass
    sim.modify_body_properties("end_effector", mass=0.5)
    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main() 