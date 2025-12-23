---
layout: home
title: Fundamentals of Robot Control
permalink: /
---

# Надежное управление в режиме скользящего режима для роботизированных манипуляторов

Фролова Анастасия Ивановна `a.frolova@innopolis.university`

В документе приведена документация к коду (final задание по курсу контрола), разбор некоторых моментов кода, графики, выводы, пояснения а также инструкция по запуску.

## Подготовка к запуску проекта
### Запуск программы
Открываем Anaconda Prompt. Оттуда идет запуск всех команд.

`conda activate robot_env` - активируем виртуальное окружение

`cd D:\control\forc\hw\mujoco_template` - путь к папке

`conda install pinocchio -c conda-forge` - установка conda, т.к. иногда появляется ошибка при перезапуске;

### Основная структура проекта

Указаны не все вложенные папки и файлы. Весь код находится в репозитории по адресу:

https://github.com/kicchhi/control/tree/main

Где находятся файлы кода к заданию отражено в схеме. Далее в отчете даются пояснения, к какому фрагменту задания относится тот или иной файл.

```bash
└── FORC
    └── hw
        └── mujoco_template
            ├── 1_ID_controller.py
            ├── 2_ID_friction.py
            ├── 3_SMC.py
            ├── 4_SMC_boundary_layer.py
            └── README.md
```

### Для быстрого коммита

`git add .`

`git commit -m "Description"` - коммит с описанием

`git pull origin main` - если в удаленном репозитории были изменения

`git push origin main`

## Assignment Tasks

1. **[40 points] Inverse Dynamics Controller**  
   - Implement Inverse Dynamics controller  

2. **[40 points] Sliding Mode Controller**  
   - Modify the UR5 robot model to include:  
     - Additional end-effector mass  
     - Joint damping coefficients  
     - Coulomb friction  
   - Implement sliding mode controller  
   - Compare ID and Sliding Mode  

3. **[20 points] Boundary Layer Implementation**  
   - Analyze the chattering phenomenon:  
     - Causes and practical implications  
     - Boundary layer modification for smoothing  
   - Evaluate performance with varying boundary layer thicknesses \(\Phi\)  
   - Analyze the robustness-chattering trade-off 

## Общая теория по задаче

Динамику манипулятора с учётом неопределённостей можно выразить следующим образом:

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) + D\dot{q} + F_c(\dot{q}) = \tau
$$

где:

- $M(q)$, $\hat{M}(q)$ — Неопределённая и оценочная матрицы масс  
- $C(q, \dot{q})$, $\hat{C}(q, \dot{q}^*)$ — Неопределённые и оценочные члены Кориолиса/центробежные члены  
- $g(q)$, $\hat{g}(q)$ — Неопределённый и оценочный векторы гравитации  
- $D$, $\hat{D}$ — Неизвестная и оценочная диагональные матрицы вязкого трения  
- $F_c$, $\hat{F}_c$ — Неизвестный и оценочный векторы кулоновского трения  
- $\tau$ — Управляющее воздействие

## 1. Inverse Dynamics Controller

`1_ID_controller.py`

За основу взят файл `01_joint_space.py`

Этот код выполняет функции управления роботом в пространстве суставов joint space с использованием PD-регулятора и учетом динамики.

Основные изменения:

```python
# u = kp * (q0 - q) + kd * (v0-dq), 
# q0, v0 - желаемые положение и скорость
# желаемая скорость равна нулю, поэтому пишем без v0
u = kp * (q0 - q) - kd * dq

# уравнение с компенсацией динамики
tau = M@u + nle
```

Результат работы в видео `logs/videos/1_ID_controller.mp4`

---

## 2. Sliding Mode Controller

`2_ID_friction.py`

Теперь модифицируем модель робота UR5, добавив дополнительную массу конечного эффектора, коэффициенты демпфирования соединений, и кулоновское трение.

Сделать это можно в xml файле напрямую, либо написав несколько строчек кода:

```python
# Set joint damping coefficients
damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
sim.set_joint_damping(damping)

# Set joint friction coefficients
friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
sim.set_joint_friction(friction)

# Modify end-effector mass
sim.modify_body_properties("end_effector", mass=4)
```

Результат работы в видео `logs/videos/2_ID_friction.mp4`

Можно заметить, что при добавлении дополнительных параметров манипулятор не может придти в конечное положение при использовании ID контроллера.

Для того, чтобы все работало исправно, нужно переписать контроллер так, чтобы эти воздействия компенсировались.

___

`3_SMC.py`

Контроллер создаёт скользящую поверхность s, которая обеспечивает робастность к неопределённостям.

Система взята из лекции 6:

$$
\begin{cases} 
u = \hat{M}(q)v + \hat{c}(q, \dot{q}) + \hat{g}(q) \\ 
v = \ddot{q}_d + \Lambda \dot{e} + v_s \\ 
v_s = \rho \frac{s}{\|s\|} \\ 
s = \dot{e} + \Lambda e 
\end{cases}
$$

Существенное отличие между материалом лекции и написанным кодом в вычислении параметра v_s (добавочный управляющий сигнал).

**В лекции**:
Умножение на М-1 нужно для попытки скомпенсировать инерционные свойства робота напрямую. Если сустав тяжелый, для его разгона нужен большой момент. Обратная матрица здесь "нормализует" управление каждым суставом. Но при таком вычислении v_s, значение получится очень маленьким, таким, что его не хватит чтобы сдвинуть робот с места. 

```python
# 1 вариант
rho = (k / sigma_max) * M_inv
v_s = rho @ s / s_norm
```

**В коде**:
Так как первый вариант часто приводит к недостаточному управлению, было принято некоторое упрощение:

```python
# 2 вариант
K_robust = 80.0
v_s = (K_robust / s_norm) * s
```

Установка корректного параметра K_robust позволяет в полной мере добиться того, чтобы робот не просто сдвинулся с места, но и дошел до конечного заданного положения.

Результат работы в видео `logs/videos/3_SMC.mp4`

Можно заметить, что управление стало немного лучше, и робот старается приблизиться к конечной точке.

## 3. Boundary Layer Implementation

`4_SMC_boundary_layer.py`

Код реализует модифицированный SMC с boundary layer.

$$
\mathbf{v}_s = 
\begin{cases} 
\rho \frac{\mathbf{s}}{\|\mathbf{s}\|}, & \|\mathbf{s}\| > \epsilon \\ 
\rho \frac{\mathbf{s}}{\epsilon}, & \|\mathbf{s}\| \leq \epsilon 
\end{cases}
$$

Вне boundary layer (когда норма s большая):
```
if s_norm > epsilon:
    v_s = (K_robust / s_norm) * s
```
Это дает сигнал постоянной амплитуды K_robust, который "толкает" систему к sliding surface.

Внутри boundary layer (когда норма s малая):
```
else:
    v_s = (K_robust / epsilon) * s
```
Здесь управление становится пропорциональным ошибке, что устраняет "дрожание" (chattering).

## Список источников

- [Lecture 4: Introduction to Nonlinear Control](https://github.com/simeon-ned/forc/blob/master/_sort/04_mech_feedback_linearization/04_feedback_linearization.pdf)
- [Lecture 6: Uncertainty, Sliding Mode, and Robust Control](https://github.com/simeon-ned/forc/blob/master/_sort/06_sliding_robust/06_sliding_mode.pdf)
- [f]()