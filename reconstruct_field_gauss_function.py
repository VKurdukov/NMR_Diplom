import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import lsq_linear
import os

# === Параметры ===
filename = r"test_data\FieldSweep 20.00K.txt"
BL = 0.72525

Bloc_min = 1e-6
Bloc_max = 0.2
num_Bloc = 70

use_nonneg = True
penalty_order = 1
auto_lambda = False
lambda_range = np.logspace(-10, 10, 1000)


# === Папка ===
output_dir = "final_gauss_data"
os.makedirs(output_dir, exist_ok=True)

# === Загрузка ===
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

print("Точек:", len(B_vals))

# === Ядро K ===
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
# 🔥 ГЛАВНОЕ: свёртка по B (правильная физика)
# =========================================================

x0_gauss = 0.008703
sigma_gauss = 0.003640

G_B = np.zeros((len(B_vals), len(B_vals)))

for i in range(len(B_vals)):
    for j in range(len(B_vals)):
        delta = B_vals[i] - B_vals[j]
        G_B[i, j] = np.exp(-0.5 * (delta / sigma_gauss)**2)

# нормировка (очень важно!)
G_B /= np.sum(G_B, axis=1, keepdims=True)

# итоговое ядро
K_eff = G_B @ K

# =========================================================

# === добавляем фон ===
K_ext = np.hstack([K_eff, np.ones((len(B_vals), 1))])

# === регуляризация ===
if penalty_order == 2:
    D = (np.diag(np.ones(num_Bloc-1), -1)
         - 2*np.diag(np.ones(num_Bloc), 0)
         + np.diag(np.ones(num_Bloc-1), 1))
elif penalty_order == 1:
    D = np.diff(np.eye(num_Bloc), axis=0)

D_ext = np.zeros((D.shape[0], num_Bloc + 1))
D_ext[:, :num_Bloc] = D

# === решение ===
def solve_with_lambda(lambda_val):
    sqrt_lambda = np.sqrt(lambda_val)

    A_aug = np.vstack([K_ext, sqrt_lambda * D_ext])
    b_aug = np.hstack([g_vals, np.zeros(D_ext.shape[0])])

    lb = np.zeros(num_Bloc + 1)
    ub = np.full(num_Bloc + 1, np.inf)

    lb[-1] = -1e20  # фон свободный

    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub))
    return res.x

# === GCV ===
def compute_gcv(lambda_val):
    sol = solve_with_lambda(lambda_val)
    g_pred = K_ext @ sol
    residual = g_vals - g_pred

    U, s, _ = np.linalg.svd(K_eff, full_matrices=False)
    trace_H = np.sum(s**2 / (s**2 + lambda_val))

    N = len(g_vals)
    return np.linalg.norm(residual)**2 / (N - trace_H)**2

def find_lambda():
    gcv_vals = []
    for lam in lambda_range:
        try:
            gcv_vals.append(compute_gcv(lam))
        except:
            gcv_vals.append(np.inf)

    gcv_vals = np.array(gcv_vals)
    idx = np.argmin(gcv_vals)

    plt.semilogx(lambda_range, gcv_vals)
    plt.scatter(lambda_range[idx], gcv_vals[idx], c='r')
    plt.grid()
    plt.title("GCV")
    plt.show()

    return lambda_range[idx]

# === основной запуск ===
if auto_lambda:
    lambda_reg = find_lambda()

sol = solve_with_lambda(lambda_reg)

f_sol = sol[:-1]
bg = sol[-1]

g_rec = K_ext @ sol

# === вывод ===
print("\n===== РЕЗУЛЬТАТ =====")
print("λ =", lambda_reg)
print("фон =", bg)

residual = np.linalg.norm(g_vals - g_rec)
print("ошибка =", residual)
print("rel =", residual / np.linalg.norm(g_vals))

# === графики ===
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(Bloc_vals, f_sol)
plt.title("f(B_loc)")
plt.grid()

plt.subplot(1,2,2)
plt.plot(B_vals, g_vals, label='g')
plt.plot(B_vals, g_rec, label='g_rec')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


base_name = os.path.basename(filename)
temp_label = base_name.replace("FieldSweep ", "").replace(".txt", "")

f_output_filename = os.path.join(output_dir, f"{temp_label}_f_Bloc.txt")

np.savetxt(
    f_output_filename,
    np.column_stack((Bloc_vals, f_sol)),
    fmt='%.8e',
    delimiter='\t',
    header="B_loc(T)\tf(B_loc)",
    comments=''
)

print(f"Сохранено: {f_output_filename}")