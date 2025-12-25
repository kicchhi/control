"""Сравнение всех 4 контроллеров - итоговый анализ."""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

def load_controller_data(name):
    """Загрузка данных контроллера."""
    try:
        data = np.load(f'logs/data/{name}_data.npz', allow_pickle=True)
        return {
            'time': data['time'],
            'q_error': data['q_error'],
            'tau': data['tau'],
            'name': name
        }
    except FileNotFoundError:
        print(f"⚠️  Данные для {name} не найдены. Запустите сначала {name}.py")
        return None

def compare_all_controllers():
    """Сравнение всех 4 контроллеров."""
    
    # Загружаем данные всех контроллеров
    controllers = [
        load_controller_data('1_ID_CONTROLLER'),
        load_controller_data('2_ID_FRICTION'), 
        load_controller_data('3_SMC'),
        load_controller_data('4_SMC_BOUNDARY_LAYER')
    ]
    
    # Убираем None
    controllers = [c for c in controllers if c is not None]
    
    if len(controllers) < 2:
        print("Нужно как минимум 2 контроллера для сравнения")
        return
    
    # Создаем фигуру для сравнения
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('COMPARATIVE ANALYSIS: ID vs SMC Controllers', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Цвета для разных контроллеров
    colors = ['blue', 'green', 'red', 'purple']
    styles = ['-', '--', '-.', ':']
    
    # 1. Суммарная ошибка по времени
    ax1 = plt.subplot(2, 3, 1)
    for idx, ctrl in enumerate(controllers):
        total_error = np.sqrt(np.sum(ctrl['q_error']**2, axis=1))
        ax1.plot(ctrl['time'], total_error, 
                color=colors[idx], linestyle=styles[idx],
                linewidth=2, label=ctrl['name'], alpha=0.8)
    
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Total Position Error [rad]', fontsize=12)
    ax1.set_title('Total Position Error Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 2. Суммарный контрольный момент
    ax2 = plt.subplot(2, 3, 2)
    for idx, ctrl in enumerate(controllers):
        total_torque = np.sqrt(np.sum(ctrl['tau']**2, axis=1))
        ax2.plot(ctrl['time'], total_torque,
                color=colors[idx], linestyle=styles[idx],
                linewidth=2, label=ctrl['name'], alpha=0.8)
    
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Total Torque [Nm]', fontsize=12)
    ax2.set_title('Control Effort Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # 3. Ошибки по каждому суставу в конце симуляции
    ax3 = plt.subplot(2, 3, 3)
    joint_labels = [f'J{i+1}' for i in range(6)]
    x_pos = np.arange(6)
    width = 0.15
    
    for idx, ctrl in enumerate(controllers):
        final_errors = np.abs(ctrl['q_error'][-1, :])
        ax3.bar(x_pos + idx*width - width*1.5, final_errors, 
               width, color=colors[idx], alpha=0.7, label=ctrl['name'])
    
    ax3.set_xlabel('Joint Number', fontsize=12)
    ax3.set_ylabel('Final Position Error [rad]', fontsize=12)
    ax3.set_title('Final Joint Errors', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(joint_labels)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right', fontsize=9)
    
    # 4. Таблица метрик
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('tight')
    ax4.axis('off')
    
    # Рассчитываем метрики
    metrics_data = []
    headers = ['Controller', 'Settling\nTime [s]', 'Max Error\n[rad]', 'RMSE\n[rad]', 
              'Avg Torque\n[Nm]', 'Energy\n[Nm²·s]']
    
    for ctrl in controllers:
        time = ctrl['time']
        q_error = ctrl['q_error']
        tau = ctrl['tau']
        
        # Расчет метрик
        settling_idx = np.where(np.all(np.abs(q_error) < 0.01, axis=1))[0]
        settling_time = time[settling_idx[0]] if len(settling_idx) > 0 else time[-1]
        
        max_error = np.max(np.abs(q_error))
        rmse = np.sqrt(np.mean(q_error**2))
        avg_torque = np.mean(np.abs(tau))
        energy = np.trapz(np.sum(tau**2, axis=1), time)
        
        metrics_data.append([
            ctrl['name'],
            f'{settling_time:.2f}',
            f'{max_error:.3f}',
            f'{rmse:.3f}',
            f'{avg_torque:.1f}',
            f'{energy:.0f}'
        ])
    
    # Создаем таблицу
    table = ax4.table(cellText=metrics_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=['lightgray']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax4.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold', y=0.95)
    
    # 5. Распределение ошибок (гистограмма)
    ax5 = plt.subplot(2, 3, 5)
    all_errors = []
    labels = []
    
    for ctrl in controllers:
        flat_errors = np.abs(ctrl['q_error']).flatten()
        all_errors.append(flat_errors)
        labels.append(ctrl['name'])
    
    ax5.boxplot(all_errors, labels=labels, patch_artist=True,
               medianprops=dict(color='black', linewidth=2),
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax5.set_ylabel('Position Error [rad]', fontsize=12)
    ax5.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.get_xticklabels(), rotation=15)
    
    # 6. Radar chart для сравнения по критериям
    ax6 = plt.subplot(2, 3, 6, polar=True)
    
    # Нормализуем метрики (меньше = лучше, кроме robustness)
    criteria = ['Precision\n(low RMSE)', 'Speed\n(low time)', 
                'Efficiency\n(low energy)', 'Smoothness\n(low torque)', 
                'Consistency\n(low var)']
    n_criteria = len(criteria)
    angles = np.linspace(0, 2*np.pi, n_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    for idx, ctrl in enumerate(controllers):
        # Здесь нужно нормализовать значения
        # Покажем примерную схему
        values = np.random.rand(n_criteria)  # ЗАМЕНИ НА РЕАЛЬНЫЕ НОРМАЛИЗОВАННЫЕ МЕТРИКИ
        values = np.append(values, values[0])
        
        ax6.plot(angles, values, color=colors[idx], linewidth=2, 
                label=ctrl['name'], alpha=0.7)
        ax6.fill(angles, values, color=colors[idx], alpha=0.1)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(criteria, fontsize=9)
    ax6.set_ylim(0, 1)
    ax6.set_title('Multi-Criteria Comparison', fontsize=14, fontweight='bold', y=1.08)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    # Сохраняем итоговый график
    os.makedirs('logs/comparison', exist_ok=True)
    plt.savefig('logs/comparison/full_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('logs/comparison/full_comparison.pdf', bbox_inches='tight')
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"✅ Individual plots: logs/plots/")
    print(f"✅ Comparison plot: logs/comparison/full_comparison.png")
    print(f"✅ Raw data: logs/data/")
    print("="*70)
    
    plt.show()

if __name__ == "__main__":
    compare_all_controllers()
