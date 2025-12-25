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
import pinocchio as pin
import os
import mujoco

SIM_DATA = {
    'time': [],
    'q': [],
    'dq': [],
    'tau': [],
    'q_error': [],
    'q_target': [],
    's_norm': [],  # только для SMC
}

MODEL = None
DATA = None

def init_model():
    """Инициализация модели"""
    global MODEL, DATA
    if MODEL is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
        MODEL = pin.buildModelFromMJCF(xml_path)
        DATA = MODEL.createData()

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

    # Проверка достижения цели
    position_error = np.max(np.abs(q0 - q))
    velocity_error = np.max(np.abs(dq))
    
    # Критерии остановки
    POSITION_TOLERANCE = 0.01  # 0.01 rad ≈ 0.57 градуса
    VELOCITY_TOLERANCE = 0.01  # 0.01 rad/s
    
    if position_error < POSITION_TOLERANCE and velocity_error < VELOCITY_TOLERANCE:
        # Робот достиг цели
        if not hasattr(joint_controller, 'target_reached_time'):
            joint_controller.target_reached_time = t
            print(f"\n🎯 ЦЕЛЬ ДОСТИГНУТА на {t:.2f} секунде!")
            print(f"   Ошибка положения: {position_error:.4f} rad")
            print(f"   Ошибка скорости: {velocity_error:.4f} rad/s")
        
        # Можно остановить симуляцию, если долго держится цель
        if hasattr(joint_controller, 'target_reached_time'):
            if t - joint_controller.target_reached_time > 2.0:  # 2 секунды после достижения
                if sim is not None:
                    print(f"\n⏹️  Останавливаю симуляцию (цель достигнута)")
                    sim.stop()  # Если есть такой метод
                    return np.zeros(6)
    
    # Load the robot model from scene XML
    global MODEL, DATA
    
    # Инициализируем модель если еще не инициализирована
    if MODEL is None:
        init_model()
    
    # Используем уже созданные
    model = MODEL
    data = DATA

    # Compute all dynamics quantities at once
    pin.computeAllTerms(model, data, q, dq)

    # Mass matrix
    M = data.M
    # Nonlinear effects (Coriolis + gravity)
    nle = data.nle

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
    
    SIM_DATA['time'].append(t)
    SIM_DATA['q'].append(q.copy())
    SIM_DATA['dq'].append(dq.copy())
    SIM_DATA['tau'].append(tau.copy())
    SIM_DATA['q_error'].append(q0 - q)
    SIM_DATA['q_target'].append(q0.copy())

    if 's_norm' in locals():  # если переменная s_norm существует
        SIM_DATA['s_norm'].append(s_norm)

    return tau

def visualize_results(data, controller_name):
    """Визуализация результатов симуляции."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    if not data['time']:
        print("Нет данных для визуализации")
        return
    
    # Конвертируем в numpy массивы
    time = np.array(data['time'])
    q = np.array(data['q'])
    dq = np.array(data['dq'])
    tau = np.array(data['tau'])
    q_error = np.array(data['q_error'])
    q_target = np.array(data['q_target'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{controller_name} - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Ошибки положений по суставам
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']
    for i in range(6):
        ax1.plot(time, q_error[:, i], color=colors[i], linewidth=1.5, 
                label=f'Joint {i+1}', alpha=0.8)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Position Error [rad]', fontsize=12)
    ax1.set_title('Joint Position Errors', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Управляющие моменты
    for i in range(6):
        ax2.plot(time, tau[:, i], color=colors[i], linewidth=1.5, 
                alpha=0.7, label=f'Tau {i+1}')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Torque [Nm]', fontsize=12)
    ax2.set_title('Control Torques', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim([-200, 200])  # Одинаковый масштаб для сравнения
    
    # 3. ошибка
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']

    for i in range(6):
        ax3.plot(time, q_error[:, i], color=colors[i], linewidth=1.5, 
                label=f'Joint {i+1}', alpha=0.8)

    # Ограничение по X (8-12 секунд)
    ax3.set_xlim([8, 12])

    # Найти максимальное значение ошибки в диапазоне 8-12 секунд
    mask = (time >= 8) & (time <= 12)
    if np.any(mask):
        # Берем максимальное абсолютное значение
        y_max = np.max(np.abs(q_error[mask, :]))
        
        # Добавляем 10% запаса и округляем до красивого числа
        y_limit = np.ceil(y_max * 1.1 * 100) / 100  # Округлить до 0.01
        
        # Если ошибка очень маленькая, установим разумный минимум
        if y_limit < 0.005:
            y_limit = 0.005
        
        ax3.set_ylim([-y_limit, y_limit])
        
        # Добавить горизонтальные линии для лучшей читаемости
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax3.axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.3, label='±0.01 rad')
        ax3.axhline(y=-0.01, color='red', linestyle='--', linewidth=1, alpha=0.3)
        
    else:
        # Если нет данных в диапазоне 8-12 секунд
        ax3.set_ylim([-0.01, 0.01])

    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_ylabel('Position Error [rad]', fontsize=12)
    ax3.set_title('Joint Position Errors (8-12 seconds, zoomed)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)  # Уменьшил шрифт легенды
    
    plt.tight_layout()
    
    # Сохраняем график
    import os
    os.makedirs('logs/plots', exist_ok=True)
    plt.savefig(f'logs/plots/{controller_name}_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'logs/plots/{controller_name}_performance.pdf', bbox_inches='tight')
    
    plt.show()

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning real-time joint space control...")
    sim = Simulator(
        # xml_path="scene.xml",
        xml_path="./robots/universal_robots_ur5e/scene.xml",
        record_video=False,
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
    sim.run(time_limit=12.0)
    
    # ДОБАВЛЯЕМ ПОСЛЕ ЗАПУСКА СИМУЛЯЦИИ:
    
    # 1. Определяем имя контроллера из имени файла
    import os
    controller_name = os.path.basename(__file__).replace('.py', '').upper()
    
    # 2. Визуализируем результаты
    visualize_results(SIM_DATA, controller_name)
    
    # 3. Сохраняем сырые данные для будущего сравнения
    os.makedirs('logs/data', exist_ok=True)
    np.savez_compressed(
        f'logs/data/{controller_name}_data.npz',
        time=np.array(SIM_DATA['time']),
        q=np.array(SIM_DATA['q']),
        dq=np.array(SIM_DATA['dq']),
        tau=np.array(SIM_DATA['tau']),
        q_error=np.array(SIM_DATA['q_error'])
    )
    
    print(f"\nДанные сохранены в logs/data/{controller_name}_data.npz")
    print(f"Графики сохранены в logs/plots/{controller_name}_performance.png")

if __name__ == "__main__":
    main() 