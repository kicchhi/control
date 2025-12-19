import roboticstoolbox as rtb
import numpy as np
from pathlib import Path

# 1. Загрузи свой MJCF/URDF файл
urdf_path = Path("robots/universal_robots_ur5e/assets/scene.xml")

# 2. Создай модель робота из файла
robot = rtb.ERobot.URDF(
    filename=str(urdf_path),
    tld=urdf_path.parent  # базовая директория для мешей
)

# 3. Выведи информацию
print(f"Робот: {robot.name}")
print(f"Число суставов: {robot.n}")
print(f"Суставы: {[joint.name for joint in robot.links if joint.isjoint]}")

# 4. Прямая кинематика
q_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
T = robot.fkine(q_home)
print(f"\nПозиция конечного эффектора (home):")
print(f"  x: {T.t[0]:.3f}, y: {T.t[1]:.3f}, z: {T.t[2]:.3f}")

# 5. Матрица инерции
M = robot.inertia(q_home)
print(f"\nМатрица инерции M (6x6):")
print(f"  Размер: {M.shape}")
print(f"  Диагональ: {np.diag(M)}")

# 6. Гравитационные силы
g = robot.gravload(q_home)
print(f"\nГравитационные силы g:")
print(f"  {g}")