import numpy as np
from simulator import Simulator
from pathlib import Path
import pinocchio as pin
import os

MODEL = None
DATA = None

def init_pinocchio():
    """Инициализация модели Pinocchio (один раз при старте)."""
    global MODEL, DATA
    if MODEL is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
        MODEL = pin.buildModelFromMJCF(xml_path)
        DATA = MODEL.createData()
        print(f"[INFO] Модель Pinocchio загружена (nq={MODEL.nq}).")
    return MODEL, DATA

def sliding_mode_controller(q: np.ndarray, dq: np.ndarray, t: float, sim=None) -> np.ndarray:
    """
    Реализация Robust Sliding Mode Controller по лекции 6.
    Формулы:
        tau = M̂(q) * v + Ĉ(q, dq) + ĝ(q)
        v = q̈_d + Λ*(q̇_d - dq) + v_s
        v_s = ρ * s / ||s||  (или sat(s/ε) для boundary layer)
        s = (q̇_d - dq) + Λ*(q_d - q)
    """
    # 1. ЦЕЛЕВЫЕ ЗНАЧЕНИЯ (REGULATION)
    q_d = np.array([-1.4, -1.3, 1.0, 0.0, 0.0, 0.0])
    dq_d = np.zeros(6)
    ddq_d = np.zeros(6)

    # 2. ПОЛУЧЕНИЕ ДИНАМИКИ (ОЦЕНКИ МОДЕЛИ)
    model, data = init_pinocchio()
    pin.computeAllTerms(model, data, q, dq)
    M_hat = data.M    # M̂(q) - оценка матрицы инерции
    nle_hat = data.nle # Ĉ(q,dq) + ĝ(q) - оценка нелинейных эффектов

    # 3. НАСТРОЙКИ КОНТРОЛЛЕРА (ПОДБИРАЙ!)
    # Λ задает динамику на скользящей поверхности (s=0 -> dq = -Λ*e)
    Lambda = np.diag([3.5, 3.5, 2.5, 2.0, 2.0, 1.5])  # Начни с этих значений

    # Коэффициент робастного управления (компенсирует w)
    K_robust = 80.0  # Начни с 80, увеличивай если не хватает, уменьшай если чаттеринг

    # Толщина граничного слоя (для boundary layer, против чаттеринга)
    epsilon = 0.08

    # 4. ВЫЧИСЛЕНИЕ СКОЛЬЗЯЩЕЙ ПОВЕРХНОСТИ
    e = q_d - q          # Вектор ошибки по положению
    de = dq_d - dq       # Вектор ошибки по скорости
    s = de + Lambda @ e  # Скользящая поверхность s

    s_norm = np.linalg.norm(s)

    v_s = (K_robust / s_norm) * s

    # 5. ВЫЧИСЛЕНИЕ РАЗРЫВНОГО УПРАВЛЕНИЯ v_s (с boundary layer)
    # Это реализация saturation function: v_s = ρ * sat(s/ε)
    # if s_norm > epsilon:
    #     v_s = (K_robust / s_norm) * s
    # else:
    #     v_s = (K_robust / epsilon) * s

    # 6. ВСПОМОГАТЕЛЬНЫЙ СИГНАЛ v И ОБЩЕЕ УПРАВЛЕНИЕ
    v = ddq_d + Lambda @ de + v_s  # v = q̈_d + Λ*(q̇_d - dq) + v_s
    tau = M_hat @ v + nle_hat      # tau = M̂(q)*v + Ĉ(q,dq) + ĝ(q)

    # 7. ОГРАНИЧЕНИЕ МОМЕНТА (ВАЖНО ДЛЯ СТАБИЛЬНОСТИ!)
    tau_limits = np.array([150, 150, 150, 50, 50, 50])
    tau = np.clip(tau, -tau_limits, tau_limits)

    # 8. ДИАГНОСТИКА (логируем каждые N шагов)
    if not hasattr(sliding_mode_controller, 'step'):
        sliding_mode_controller.step = 0
    sliding_mode_controller.step += 1

    if sliding_mode_controller.step % 100 == 0:
        pos_error = np.linalg.norm(e)
        print(f"[SMC] t={t:5.2f}s | pos err: {pos_error:7.4f} rad | s norm: {s_norm:7.4f} | tau max: {np.max(np.abs(tau)):6.2f} Nm")

    return tau

def main():
    """Основная функция для запуска SMC."""
    Path("logs/videos").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("Запуск Sliding Mode Controller (SMC) для UR5e")
    print("="*50)

    sim = Simulator(
        xml_path="./robots/universal_robots_ur5e/scene2.xml",
        record_video=True,
        video_path="logs/videos/ur5e_smc.mp4",
        width=1280,
        height=720
    )

    # НАСТРОЙКА НЕОПРЕДЕЛЕННОСТЕЙ (как в задании)
    print("\n[INFO] Устанавливаю параметры неопределенностей...")
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    sim.set_joint_damping(damping)
    sim.set_joint_friction(friction)
    sim.modify_body_properties("end_effector", mass=4.0)  # Измененная масса

    sim.set_controller(sliding_mode_controller)
    sim.run(time_limit=10.0)  # Время симуляции

    print("\n[INFO] Симуляция завершена.")

if __name__ == "__main__":
    main()