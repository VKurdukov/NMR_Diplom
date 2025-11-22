import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import lsq_linear
import os

# Параметры
filename = r"test\FieldSweep 25.00K.txt"
BL = 0.72525
Bloc_min = 1e-6
Bloc_max = 0.2
num_Bloc = 70
use_nonneg = True
penalty_order = 2
auto_lambda = True
lambda_range = np.logspace(0, 10, 100)

# Создаём папку для результатов
output_dir = "final"
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
else:
    print(len(B_vals))


try:
    base_name = os.path.basename(filename)
    temp_label = base_name.replace("FieldSweep ", "").replace(".txt", "")
    output_filename = os.path.join(output_dir, f"{temp_label}_reconstructed.txt")
except Exception:
    output_filename = os.path.join(output_dir, "reconstructed_spectrum.txt")

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

# Матрица штрафа
if penalty_order == 2:
    D = (np.diag(np.ones(num_Bloc-1), -1)
         - 2*np.diag(np.ones(num_Bloc), 0)
         + np.diag(np.ones(num_Bloc-1), 1))
elif penalty_order == 1:
    D = np.diff(np.eye(num_Bloc), axis=0)
else:
    raise ValueError("penalty_order = 1 или 2")

D_ext = np.zeros((D.shape[0], num_Bloc + 1))
D_ext[:, :num_Bloc] = D

# GCV
def solve_with_lambda(lambda_val):
    sqrt_lambda = np.sqrt(lambda_val)
    A_top, b_top = K_ext, g_vals
    A_bot, b_bot = sqrt_lambda * D_ext, np.zeros(D_ext.shape[0])
    A_aug = np.vstack([A_top, A_bot])
    b_aug = np.hstack([b_top, b_bot])

    if use_nonneg:
        lb = np.zeros(num_Bloc + 1)
        ub = np.full(num_Bloc + 1, np.inf)
        lb[-1] = -1e20
    else:
        lb = np.full(num_Bloc + 1, -1e20)
        ub = np.full(num_Bloc + 1, 1e20)

    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), lsmr_tol='auto')
    return res.x, res.cost

def compute_gcv(lambda_val):
    sol, _ = solve_with_lambda(lambda_val)
    g_pred = K_ext @ sol
    residual = g_vals - g_pred
    U, s, Vt = np.linalg.svd(K, full_matrices=False)
    trace_H = np.sum(s**2 / (s**2 + lambda_val))
    N = len(g_vals)
    gcv = np.linalg.norm(residual)**2 / (N - trace_H)**2
    return gcv

def find_lambda_gcv():
    gcv_vals = []
    for lam in lambda_range:
        try:
            gcv = compute_gcv(lam)
            gcv_vals.append(gcv)
        except np.linalg.LinAlgError:
            gcv_vals.append(np.inf)

    gcv_vals = np.array(gcv_vals)
    idx_opt = np.argmin(gcv_vals)
    lambda_opt = lambda_range[idx_opt]

    plt.figure(figsize=(6, 4))
    plt.semilogx(lambda_range, gcv_vals, '-o', label='GCV(λ)')
    plt.scatter(lambda_opt, gcv_vals[idx_opt], color='r', label=f'λ_opt={lambda_opt:.2e}')
    plt.xlabel('λ')
    plt.ylabel('GCV(λ)')
    plt.title('Generalized Cross-Validation')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Оптимальное λ по GCV = {lambda_opt:.3e}")
    return lambda_opt

# Основное решение
if auto_lambda:
    lambda_reg = find_lambda_gcv()

sqrt_lambda = np.sqrt(lambda_reg)
A_top, b_top = K_ext, g_vals
A_bot, b_bot = sqrt_lambda * D_ext, np.zeros(D_ext.shape[0])
A_aug = np.vstack([A_top, A_bot])
b_aug = np.hstack([b_top, b_bot])

if use_nonneg:
    lb = np.zeros(num_Bloc + 1)
    ub = np.full(num_Bloc + 1, np.inf)
    lb[-1] = -1e20
else:
    lb = np.full(num_Bloc + 1, -1e20)
    ub = np.full(num_Bloc + 1, 1e20)

res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
sol = res.x
f_sol = sol[:-1]
bg = sol[-1]

print("\n==============================")
print(f"Фон: {bg:.6f}")
print(f"λ (рег.) = {lambda_reg:.3e}")
print("min(f) = ", f_sol.min(), "max(f) = ", f_sol.max())
g_rec = K_ext @ sol
residual = g_vals - g_rec
print("Норма невязки:", np.linalg.norm(residual), "Rel err:", np.linalg.norm(residual)/np.linalg.norm(g_vals))

#СОХРАНЕНИЕ РАСПРЕДЕЛЕНИЯ ЛОКАЛЬНЫХ ПОЛЕЙ f(B_loc)
f_output_filename = os.path.join(output_dir, f"{temp_label}_f_Bloc.txt")

f_data = np.column_stack((Bloc_vals, f_sol))
np.savetxt(f_output_filename, f_data, fmt='%.8e', delimiter='\t',
           header=f"B_loc(T)\tf(B_loc)\nFile: {os.path.basename(filename)}\nlambda = {lambda_reg:.3e}",
           comments='')

print(f"Распределение f(B_loc) сохранено: {f_output_filename}")

# Визуализация
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(Bloc_vals, f_sol, '-o')
plt.title(f'f(B_loc), λ={lambda_reg:.2e}')
plt.xlabel('B_loc')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(B_vals, g_vals, label='g')
plt.plot(B_vals, g_rec, label='g_rec', lw=2)
plt.legend()
plt.xlabel('B')
plt.grid(True)

plt.tight_layout()
plt.show()