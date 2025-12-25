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

SIM_DATA = {
    'time': [],
    'q': [],
    'dq': [],
    'tau': [],
    'q_error': [],
    'q_target': [],
    's_norm': [],  # только для SMC
    'in_boundary': []  # только для SMC с boundary layer
}

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float, sim=None, epsilon=1.0) -> np.ndarray:
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

    # 1. Матрица Λ (положительно определенная)
    Lambda = np.diag([15 for i in range(6)])
    # Lambda = np.diag([2.0, 2.0, 1.5, 1.0, 1.0, 0.8])  # Гораздо меньше!

    # Ошибки
    e = q0 - q  # Ошибка положения
    de = dq_d - dq  # Ошибка скорости
    
    # Скользящая поверхность: s = de + Λ·e
    s = de + Lambda @ e
    
    # Норма скользящей поверхности
    s_norm = np.linalg.norm(s)
    
    K_robust = 80.0

    if s_norm > epsilon:
        v_s = (K_robust / s_norm ) * s
    else:
        v_s = (K_robust / epsilon) * s
    
    # После вычисления v_s добавь:
    if not hasattr(joint_controller, 'counter'):
        joint_controller.counter = 0
        joint_controller.in_boundary = 0

    joint_controller.counter += 1
    if s_norm <= epsilon:
        joint_controller.in_boundary += 1

    if joint_controller.counter % 100 == 0:
        boundary_percent = (joint_controller.in_boundary / joint_controller.counter) * 100
        print(f"В boundary layer: {boundary_percent:.1f}% времени")

    # Вспомогательный сигнал v
    v = ddq_d + Lambda @ de + v_s
    
    # Основное управление: u = M·v + Ĉ + ĝ
    tau = M @ v + nle
    
    SIM_DATA['time'].append(t)
    SIM_DATA['q'].append(q.copy())
    SIM_DATA['dq'].append(dq.copy())
    SIM_DATA['tau'].append(tau.copy())
    SIM_DATA['q_error'].append(q0 - q)
    SIM_DATA['q_target'].append(q0.copy())

    if 's_norm' in locals():  # если переменная s_norm существует
        SIM_DATA['s_norm'].append(s_norm)

    if 'epsilon' in locals():
        SIM_DATA['in_boundary'].append(1 if s_norm <= epsilon else 0)

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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # 3 графика в ряд
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
    
    # 3. Фазовый портрет для 1-го сустава (пример)
    ax3.plot(q[:, 0], dq[:, 0], 'b-', linewidth=1.5, alpha=0.7)
    ax3.scatter(q_target[0, 0], 0, color='red', s=100, marker='*', 
               label='Target', zorder=5)
    ax3.set_xlabel('Joint 1 Position [rad]', fontsize=12)
    ax3.set_ylabel('Joint 1 Velocity [rad/s]', fontsize=12)
    ax3.set_title('Phase Portrait (Joint 1)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 3. Ошибки положений (зум 8-12 секунд)
    for i in range(6):
        ax3.plot(time, q_error[:, i], color=colors[i], linewidth=1.5, 
                label=f'Joint {i+1}', alpha=0.8)
    
    # Ограничение по X (8-12 секунд)
    ax3.set_xlim([8, 12])
    
    # Автоматический масштаб по Y
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
        
        print(f"✅ Масштаб Y (8-12s): ±{y_limit:.3f} rad")
    else:
        # Если нет данных в диапазоне 8-12 секунд
        ax3.set_ylim([-0.01, 0.01])
        print("⚠️  Нет данных в диапазоне 8-12 секунд")
    
    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_ylabel('Position Error [rad]', fontsize=12)
    ax3.set_title('Joint Position Errors (8-12 seconds, zoomed)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    # Сохраняем график
    import os
    os.makedirs('logs/plots', exist_ok=True)
    plt.savefig(f'logs/plots/{controller_name}_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'logs/plots/{controller_name}_performance.pdf', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

    if 'in_boundary' in data and data['in_boundary']:
        boundary_percent = np.mean(data['in_boundary']) * 100
        print(f"📊 {controller_name}: {boundary_percent:.1f}% времени в boundary layer")

def main():
    # Создаем список значений epsilon для экспериментов
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Экспериментируем с разными значениями
    
    # Будем хранить результаты для каждого epsilon
    all_results = {}
    
    for epsilon in epsilon_values:
        print(f"\n{'='*60}")
        print(f"Запуск с epsilon = {epsilon}")
        print(f"{'='*60}")
        
        # Очищаем данные перед каждым запуском
        global SIM_DATA
        SIM_DATA = {
            'time': [],
            'q': [],
            'dq': [],
            'tau': [],
            'q_error': [],
            'q_target': [],
            's_norm': [],
            'in_boundary': []
        }
        
        # Сбрасываем счетчики контроллера
        if hasattr(joint_controller, 'counter'):
            delattr(joint_controller, 'counter')
        if hasattr(joint_controller, 'in_boundary'):
            delattr(joint_controller, 'in_boundary')
        
        # Создаем симулятор (только для epsilon=1.0 записываем видео)
        record_video = (epsilon == 1.0)  # Видео только для epsilon=1.0
        
        print(f"Запись видео: {'ВКЛ' if record_video else 'ВЫКЛ'}")
        
        # ИСПРАВЛЕНО: Создаем симулятор с видео или без
        if record_video:
            sim = Simulator(
                xml_path="./robots/universal_robots_ur5e/scene.xml",
                record_video=True,
                video_path=f"logs/videos/SMC_epsilon_{epsilon}.mp4",
                width=1920,
                height=1080
            )
        else:
            sim = Simulator(
                xml_path="./robots/universal_robots_ur5e/scene.xml",
                record_video=False,  # Явно отключаем запись
                width=1920,
                height=1080
            )
        
        # Настройка параметров робота
        damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        sim.set_joint_damping(damping)
        
        friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        sim.set_joint_friction(friction)
        
        sim.modify_body_properties("end_effector", mass=0.5)
        
        # Создаем замыкание для передачи epsilon в контроллер
        def controller_with_epsilon(q, dq, t, sim=sim):
            return joint_controller(q, dq, t, sim, epsilon=epsilon)
        
        sim.set_controller(controller_with_epsilon)
        
        # Запускаем симуляцию
        sim.run(time_limit=12.0)
        
        # Сохраняем результаты для этого epsilon
        all_results[epsilon] = {
            'time': np.array(SIM_DATA['time']),
            'q': np.array(SIM_DATA['q']),
            'dq': np.array(SIM_DATA['dq']),
            'tau': np.array(SIM_DATA['tau']),
            'q_error': np.array(SIM_DATA['q_error']),
            's_norm': np.array(SIM_DATA['s_norm']) if SIM_DATA['s_norm'] else None,
            'in_boundary': np.array(SIM_DATA['in_boundary']) if SIM_DATA['in_boundary'] else None
        }
        
        controller_name = f"SMC_epsilon_{epsilon}"
        
        # Сохраняем сырые данные
        os.makedirs('logs/data', exist_ok=True)
        np.savez_compressed(
            f'logs/data/{controller_name}_data.npz',
            time=all_results[epsilon]['time'],
            q=all_results[epsilon]['q'],
            dq=all_results[epsilon]['dq'],
            tau=all_results[epsilon]['tau'],
            q_error=all_results[epsilon]['q_error'],
            s_norm=all_results[epsilon]['s_norm'],
            in_boundary=all_results[epsilon]['in_boundary']
        )
    
    # После всех симуляций создаем СРАВНИТЕЛЬНЫЕ графики
    plot_comparison(all_results)

def plot_comparison(all_results):
    """Создает графики ошибок на интервале 8-12 секунд для разных epsilon."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Создаем фигуру: 5 строк (по одной на epsilon) × 1 столбец
    fig = plt.figure(figsize=(12, 20))
    
    # Создаем сетку 5x1 (5 epsilon, 1 тип графика)
    gs = GridSpec(5, 1, figure=fig, hspace=0.35, wspace=0.2)
    
    # Цвета для разных epsilon
    epsilon_colors = ['darkblue', 'darkgreen', 'darkred', 'teal', 'purple']
    
    for row_idx, (epsilon, data) in enumerate(all_results.items()):
        # Создаем один график на строку (интервал 8-12 секунд)
        ax = fig.add_subplot(gs[row_idx, 0])
        
        # Рисуем ВСЕ 6 суставов на одном графике
        for i in range(6):
            ax.plot(data['time'], data['q_error'][:, i], 
                   linewidth=1.5, alpha=0.7, label=f'Joint {i+1}')
        
        # Ограничение по X (8-12 секунд)
        ax.set_xlim([8, 12])
        
        # Найти максимальное значение по всем epsilon
        global_y_max = 0
        for data in all_results.values():
            mask = (data['time'] >= 8) & (data['time'] <= 12)
            if np.any(mask):
                y_max = np.max(np.abs(data['q_error'][mask, :]))
                global_y_max = max(global_y_max, y_max)

        # Выбрать одинаковый предел для всех
        y_limit = np.ceil(global_y_max * 1.2 * 100) / 100
        if y_limit < 0.005:
            y_limit = 0.005

        # Внутри цикла для каждого epsilon:
        ax.set_ylim([-y_limit, y_limit])
        
        # Горизонтальные линии для ориентира
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(y=0.01, color='red', linestyle='--', linewidth=1.0, alpha=0.4, label='±0.01 rad')
        ax.axhline(y=-0.01, color='red', linestyle='--', linewidth=1.0, alpha=0.4)
        
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Position Error [rad]', fontsize=11)
        ax.set_title(f'ε = {epsilon}: Joint Position Errors (8-12 seconds)', 
                    fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3)
        
        # Легенда только для первого графика
        if row_idx == 0:
            ax.legend(loc='upper right', fontsize=9, ncol=3)
    
    # Общий заголовок
    plt.suptitle('SMC Controller: Joint Position Errors (8-12 seconds zoomed)\nComparison for Different Boundary Layer Thickness (ε)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Сохраняем
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs('logs/plots', exist_ok=True)
    plt.savefig('logs/plots/SMC_epsilon_8_12_seconds.png', dpi=300, bbox_inches='tight')
    plt.savefig('logs/plots/SMC_epsilon_8_12_seconds.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Дополнительно: статистика в консоли
    print(f"\n{'='*70}")
    print("SUMMARY: Position Errors on 8-12 seconds interval")
    print(f"{'='*70}")
    
    for epsilon, data in all_results.items():
        mask = (data['time'] >= 8) & (data['time'] <= 12)
        if np.any(mask):
            errors = data['q_error'][mask, :]
            max_error = np.max(np.abs(errors))
            mean_error = np.mean(np.abs(errors))
            std_error = np.std(errors)
            
            if data['in_boundary'] is not None:
                boundary_percent = np.mean(data['in_boundary'][mask]) * 100 if mask.any() else 0
                print(f"ε={epsilon}: "
                      f"Max error = {max_error:.4f} rad, "
                      f"Mean error = {mean_error:.4f} rad, "
                      f"Std = {std_error:.4f}, "
                      f"Boundary = {boundary_percent:.1f}%")
if __name__ == "__main__":
    main() 