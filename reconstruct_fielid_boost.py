import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import lsq_linear
import os

# Параметры
filename = r"test_data\FieldSweep 12.00K.txt"
BL = 0.72525
Bloc_min = 1e-6
Bloc_max = 0.2
num_Bloc = 70
penalty_order = 2
n_bootstrap = 2000

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

# Ядро
def K_new(B, Bloc):
    if B == 0 or Bloc == 0:
        return 0.0
    if Bloc < abs(B - BL):
        return 0.0
    return (B**2 - Bloc**2 + BL**2) / (Bloc * B**2)

Bloc_vals = np.linspace(Bloc_min, Bloc_max, num_Bloc)
K = np.zeros((len(B_vals), len(Bloc_vals)))
for i, B in enumerate(B_vals):
    for j, Bloc in enumerate(Bloc_vals):
        K[i, j] = K_new(B, Bloc)

K_ext = np.hstack([K, np.ones((len(B_vals), 1))])

# Матрица регуляризации
if penalty_order == 2:
    D = (np.diag(np.ones(num_Bloc-1), -1)
         - 2*np.diag(np.ones(num_Bloc), 0)
         + np.diag(np.ones(num_Bloc-1), 1))
else:
    D = np.diff(np.eye(num_Bloc), axis=0)

D_ext = np.zeros((D.shape[0], num_Bloc + 1))
D_ext[:, :num_Bloc] = D

# Решение Тихонова
def solve_tikhonov(lambda_val, g_data):
    sqrt_lambda = np.sqrt(lambda_val)
    A_aug = np.vstack([K_ext, sqrt_lambda * D_ext])
    b_aug = np.hstack([g_data, np.zeros(D_ext.shape[0])])
    
    lb = np.zeros(num_Bloc + 1)
    ub = np.full(num_Bloc + 1, np.inf)
    lb[-1] = -1e20
    
    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), lsmr_tol='auto')
    return res.x

# Подбор λ через GCV
lambda_range = np.logspace(0, 10, 100)
best_lambda = lambda_range[0]
best_gcv = np.inf

print("Поиск оптимального λ методом GCV...")
for lam in lambda_range:
    sol = solve_tikhonov(lam, g_vals)
    g_pred = K_ext @ sol
    residual = g_vals - g_pred
    U, s, Vt = np.linalg.svd(K, full_matrices=False)
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

# Невязка
g_rec_bootstrap = K @ f_mean + bg_mean
residual_bootstrap = g_vals - g_rec_bootstrap
rel_err = np.linalg.norm(residual_bootstrap) / np.linalg.norm(g_vals)

print(f"\n=== РЕЗУЛЬТАТЫ БУТСТРЕПА ===")
print(f"Фон: {bg_mean:.6f} ± {bg_std:.6f}")
print(f"min(f) = {f_mean.min():.6f}, max(f) = {f_mean.max():.6f}")
print(f"Относительная ошибка: {rel_err*100:.2f}%")

# Сохранение ТОЛЬКО файла с распределением (3 колонки: B_loc, f, error)
f_output_filename = os.path.join(output_dir, f"{temp_label}_f_Bloc_bootstrap.txt")
f_data = np.column_stack((Bloc_vals, f_mean, f_std))
np.savetxt(f_output_filename, f_data, fmt='%.8e', delimiter='\t',
           header=f"B_loc(T)\tf(B_loc)\terror\nFile: {os.path.basename(filename)}\nMethod: Bootstrap (n={n_bootstrap})\nlambda={best_lambda:.3e}\nnoise_std={noise_std:.3e}\nrel_err={rel_err:.4f}",
           comments='')

print(f"Сохранено: {f_output_filename}")