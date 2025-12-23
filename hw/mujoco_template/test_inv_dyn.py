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
    q0 = np.array([-1.4, -1.3, 1., 0, 0, 0])
    
    # Load the robot model from scene XML
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()

    # Print basic model information
    print("\nRobot Model Info:")
    print(f"Number of DOF: {model.nq}")

    # =====================
    # Dynamics Computations
    # =====================
    # Compute all dynamics quantities at once
    pin.computeAllTerms(model, data, q, dq)

    # Mass matrix
    M = data.M
    print("\nMass Matrix:")
    print(M)

    # Nonlinear effects (Coriolis + gravity)
    nle = data.nle
    print("\nNon-Linear Effects (Coriolis + Gravity):")
    print(nle)

    # PD control law
    u = kp * (q0 - q) - kd * dq
    print(q,"<-----------")
    
    tau = M@u + nle
    # Mddq + nle = M@u + nle
    # M-1Mddq = M-1M@u
    # ddq = u
    # PD - регулятор
    # tau = kp * (q0 - q) + kd * (v0-dq), q0, v0 - желаемые положение и скорость
    # мы хотим прийти в нужное положение и остановиться, поэтому
    # желаемая скорость равна нулю, поэтому пишем так!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #tau = kp * (q0 - q) - kd * dq
    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning real-time joint space control...")
    sim = Simulator(
        # xml_path="scene.xml",
        xml_path="./robots/universal_robots_ur5e/scene.xml",
        record_video=True,
        video_path="logs/videos/test_id.mp4",
        width=1920,
        height=1080
    )
    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main() 