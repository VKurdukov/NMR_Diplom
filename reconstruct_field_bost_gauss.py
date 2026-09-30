import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import lsq_linear
import os

# Параметры
filename = r"test_data\FieldSweep 20.00K.txt"
BL = 0.72525
Bloc_min = 1e-6
Bloc_max = 0.2
num_Bloc = 70
penalty_order = 2
n_bootstrap = 500

# Параметры гаусса (собственное уширение спектра)
x0_gauss = 0.008703
sigma_gauss = 0.003640

# =========================================================
# НОВЫЕ ПАРАМЕТРЫ: ШТРАФ ЗА НЕНУЛЕВЫЕ ЗНАЧЕНИЯ НА КРАЯХ
# =========================================================
N_edge_left = 0      # Количество точек слева (b_loc → 0), которые штрафуем
N_edge_right = 65     # Количество точек справа (b_loc → 0.2), которые штрафуем
edge_penalty_weight = 10.0  # Коэффициент усиления штрафа (относительно λ)

output_dir = "bootstrap_results"
os.makedirs(output_dir, exist_ok=True)

# Загрузка данных
def load_field_data(filepath):
    B_vals, g_vals = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and re.match(r"^-?\d+(\.\d+)?$", parts[0]):
                try:
                    B_vals.append(float(parts[0]))
                    g_vals.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(B_vals), np.array(g_vals)

B_vals, g_vals = load_field_data(filename)
if len(B_vals) == 0:
    raise RuntimeError("Нет числовых данных")

try:
    base_name = os.path.basename(filename)
    temp_label = base_name.replace("FieldSweep ", "").replace(".txt", "")
except Exception:
    temp_label = "unknown"

# Ядро K (без свёртки)
def K_new(B, Bloc):
    if B == 0 or Bloc == 0:
        return 0.0
    if Bloc < abs(B - BL):
        return 0.0
    return (B**2 - Bloc**2 + BL**2) / (Bloc * B**2)

Bloc_vals = np.linspace(Bloc_min, Bloc_max, num_Bloc)
K = np.zeros((len(B_vals), num_Bloc))
for i, B in enumerate(B_vals):
    for j, Bloc in enumerate(Bloc_vals):
        K[i, j] = K_new(B, Bloc)

# =========================================================
# СВЁРТКА С ГАУССОМ (собственное уширение спектра)
# =========================================================
print("Построение матрицы гауссовой свёртки...")
G_B = np.zeros((len(B_vals), len(B_vals)))
for i in range(len(B_vals)):
    for j in range(len(B_vals)):
        delta = B_vals[i] - B_vals[j]
        G_B[i, j] = np.exp(-0.5 * (delta / sigma_gauss)**2)

G_B /= np.sum(G_B, axis=1, keepdims=True)
K_eff = G_B @ K
print("Свёртка построена.")

K_ext = np.hstack([K_eff, np.ones((len(B_vals), 1))])

# Матрица регуляризации (гладкость)
if penalty_order == 2:
    D = (np.diag(np.ones(num_Bloc-1), -1)
         - 2*np.diag(np.ones(num_Bloc), 0)
         + np.diag(np.ones(num_Bloc-1), 1))
else:
    D = np.diff(np.eye(num_Bloc), axis=0)

D_ext = np.zeros((D.shape[0], num_Bloc + 1))
D_ext[:, :num_Bloc] = D

# =========================================================
# НОВОЕ: МАТРИЦА ШТРАФА ЗА НЕНУЛЕВЫЕ ЗНАЧЕНИЯ НА КРАЯХ
# =========================================================
# Создаём матрицу, которая штрафует f_i → 0 для крайних точек
# Каждая строка соответствует одному уравнению: w * f_i = 0
edge_rows = []
for i in range(N_edge_left):
    row = np.zeros(num_Bloc + 1)
    row[i] = edge_penalty_weight  # Штраф для f[0], f[1], ..., f[N_edge_left-1]
    edge_rows.append(row)

for i in range(N_edge_right):
    row = np.zeros(num_Bloc + 1)
    row[num_Bloc - 1 - i] = edge_penalty_weight  # Штраф для f[-1], f[-2], ...
    edge_rows.append(row)

D_edge = np.array(edge_rows)  # Размер: (N_edge_left + N_edge_right) × (num_Bloc + 1)

# Объединяем матрицу гладкости и матрицу краевого штрафа
D_combined = np.vstack([D_ext, D_edge])

print(f"Добавлен краевой штраф: {N_edge_left} точек слева, {N_edge_right} точек справа")
print(f"Вес штрафа: {edge_penalty_weight}")

# Решение Тихонова (с комбинированной регуляризацией)
def solve_tikhonov(lambda_val, g_data):
    sqrt_lambda = np.sqrt(lambda_val)
    # Верхняя часть: подгонка под данные
    A_top = K_ext
    b_top = g_data
    # Нижняя часть: регуляризация (гладкость + краевой штраф)
    A_bot = sqrt_lambda * D_combined
    b_bot = np.zeros(D_combined.shape[0])
    
    A_aug = np.vstack([A_top, A_bot])
    b_aug = np.hstack([b_top, b_bot])
    
    lb = np.zeros(num_Bloc + 1)
    ub = np.full(num_Bloc + 1, np.inf)
    lb[-1] = -1e20
    
    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), lsmr_tol='auto')
    return res.x

# Подбор λ через GCV (с K_eff)
lambda_range = np.logspace(0, 10, 100)
best_lambda = lambda_range[0]
best_gcv = np.inf

print("Поиск оптимального λ методом GCV...")
for lam in lambda_range:
    sol = solve_tikhonov(lam, g_vals)
    g_pred = K_ext @ sol
    residual = g_vals - g_pred
    U, s, Vt = np.linalg.svd(K_eff, full_matrices=False)
    trace_H = np.sum(s**2 / (s**2 + lam))
    N = len(g_vals)
    gcv = np.linalg.norm(residual)**2 / (N - trace_H)**2
    if gcv < best_gcv:
        best_gcv = gcv
        best_lambda = lam

print(f"Оптимальный λ = {best_lambda:.3e}")

# Основное решение
sol_main = solve_tikhonov(best_lambda, g_vals)
f_main = sol_main[:-1]
bg_main = sol_main[-1]
g_rec_main = K_ext @ sol_main
residual_main = g_vals - g_rec_main

noise_std = np.std(np.diff(residual_main)) / np.sqrt(2)
print(f"Оценка шума σ = {noise_std:.3f}")
print(f"Относительная ошибка: {np.linalg.norm(residual_main)/np.linalg.norm(g_vals)*100:.2f}%")

# ============================================================
# БУТСТРЕП
# ============================================================
def bootstrap_iteration(seed):
    np.random.seed(seed)
    g_noisy = g_vals + noise_std * np.random.randn(len(g_vals))
    return solve_tikhonov(best_lambda, g_noisy)

print(f"\nЗапуск {n_bootstrap} бутстреп-итераций...")
solutions = []
for i in range(n_bootstrap):
    if i % 200 == 0:
        print(f"  Прогресс: {i}/{n_bootstrap}")
    solutions.append(bootstrap_iteration(i))

solutions = np.array(solutions)
f_solutions = solutions[:, :num_Bloc]
bg_solutions = solutions[:, num_Bloc]

# Статистики
f_mean = np.mean(f_solutions, axis=0)
f_std = np.std(f_solutions, axis=0)
bg_mean = np.mean(bg_solutions)
bg_std = np.std(bg_solutions)

# Невязка среднего решения
g_rec_bootstrap = K_ext @ np.append(f_mean, bg_mean)
residual_bootstrap = g_vals - g_rec_bootstrap
rel_err = np.linalg.norm(residual_bootstrap) / np.linalg.norm(g_vals)

print(f"\n=== РЕЗУЛЬТАТЫ БУТСТРЕПА ===")
print(f"Фон: {bg_mean:.6f} ± {bg_std:.6f}")
print(f"min(f) = {f_mean.min():.6f}, max(f) = {f_mean.max():.6f}")
print(f"Относительная ошибка: {rel_err*100:.2f}%")

# Проверка: значения на краях
print(f"\nПроверка краевых значений:")
print(f"  f[0:{N_edge_left}] = {f_mean[:N_edge_left]}")
print(f"  f[-{N_edge_right}:] = {f_mean[-N_edge_right:]}")

# Сохранение ТОЛЬКО файла с распределением (3 колонки: B_loc, f, error)
f_output_filename = os.path.join(output_dir, f"{temp_label}_f_Bloc_bootstrap.txt")
f_data = np.column_stack((Bloc_vals, f_mean, f_std))
np.savetxt(f_output_filename, f_data, fmt='%.8e', delimiter='\t',
           header=f"B_loc(T)\tf(B_loc)\terror\nFile: {os.path.basename(filename)}\nMethod: Bootstrap (n={n_bootstrap})\nlambda={best_lambda:.3e}\nnoise_std={noise_std:.3e}\nrel_err={rel_err:.4f}\nsigma_gauss={sigma_gauss:.6f}\nedge_penalty_left={N_edge_left}\nedge_penalty_right={N_edge_right}\nedge_weight={edge_penalty_weight}",
           comments='')

print(f"Сохранено: {f_output_filename}")

# ============================================================
# ГРАФИКИ
# ============================================================

# График 1: Сравнение распределений (Тихонов vs Bootstrap)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(Bloc_vals, f_main, 'r--', label='Tikhonov (original)', alpha=0.7, linewidth=2)
plt.plot(Bloc_vals, f_mean, 'b-', label='Bootstrap mean', linewidth=2)
plt.fill_between(Bloc_vals, f_mean - f_std, f_mean + f_std, alpha=0.3, color='blue', label='±1σ (bootstrap)')

# Подсветка краевых областей
plt.axvspan(Bloc_vals[0], Bloc_vals[N_edge_left-1], alpha=0.1, color='orange', label='Edge penalty (left)')
plt.axvspan(Bloc_vals[-N_edge_right], Bloc_vals[-1], alpha=0.1, color='orange', label='Edge penalty (right)')

plt.title(f'Local Field Distribution\n{temp_label}', fontsize=12)
plt.xlabel('B_loc (T)')
plt.ylabel('f(B_loc)')
plt.legend()
plt.grid(True)

# График 2: Сравнение спектров (эксперимент vs реконструкция)
plt.subplot(1, 2, 2)
plt.plot(B_vals, g_vals, 'k.', label='Experimental data', markersize=3)
plt.plot(B_vals, g_rec_main, 'r--', label='Tikhonov reconstruction', alpha=0.7, linewidth=2)
plt.plot(B_vals, g_rec_bootstrap, 'b-', label='Bootstrap reconstruction', linewidth=2)
plt.legend()
plt.xlabel('B (T)')
plt.ylabel('Intensity')
plt.title(f'Relative error: {rel_err*100:.2f}%\nλ = {best_lambda:.2e}', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()

# График 3: Гистограмма значений в конкретной точке (диагностика)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
mid_idx = num_Bloc // 2
plt.hist(f_solutions[:, mid_idx], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.axvline(f_mean[mid_idx], color='blue', linestyle='--', linewidth=2, label=f'Mean: {f_mean[mid_idx]:.4f}')
plt.axvline(f_mean[mid_idx] - f_std[mid_idx], color='red', linestyle=':', linewidth=1.5, label=f'±1σ')
plt.axvline(f_mean[mid_idx] + f_std[mid_idx], color='red', linestyle=':', linewidth=1.5)
plt.title(f'Distribution at B_loc = {Bloc_vals[mid_idx]:.4f} T', fontsize=11)
plt.xlabel('f(B_loc)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.5)

plt.subplot(1, 2, 2)
plt.hist(bg_solutions, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(bg_mean, color='blue', linestyle='--', linewidth=2, label=f'Mean: {bg_mean:.4f}')
plt.axvline(bg_mean - bg_std, color='red', linestyle=':', linewidth=1.5, label=f'±1σ')
plt.axvline(bg_mean + bg_std, color='red', linestyle=':', linewidth=1.5)
plt.title('Background distribution', fontsize=11)
plt.xlabel('Background')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

print("\n✅ Готово! Построено 3 графика.")