import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import lsq_linear
import os
import emcee

# Параметры
filename = r"test_data\FieldSweep 6.20K.txt"
BL = 0.72525
Bloc_min = 1e-6
Bloc_max = 0.2
num_Bloc = 40
penalty_order = 2

# MCMC параметры
n_walkers = 200  # Увеличили (больше walkers = лучше исследование)
n_steps = 5000
burn_in = 1500

output_dir = "bayesian_results"
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

# ============================================================
# ШАГ 1: Решение Тихонова
# ============================================================
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
lambda_range = np.logspace(-3, 3, 50)
best_lambda = lambda_range[0]
best_gcv = np.inf

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

print(f"Оптимальный λ (Тихонов) = {best_lambda:.3e}")

sol_tikhonov = solve_tikhonov(best_lambda, g_vals)
f_tikhonov = sol_tikhonov[:-1]
bg_tikhonov = sol_tikhonov[-1]
g_rec_tikh = K_ext @ sol_tikhonov
residual_tikh = g_vals - g_rec_tikh

# Оценка шума
noise_std = np.std(np.diff(residual_tikh)) / np.sqrt(2)
print(f"Оценка шума σ = {noise_std:.6f}")
print(f"Относительная ошибка Тихонова: {np.linalg.norm(residual_tikh)/np.linalg.norm(g_vals)*100:.2f}%")

# ============================================================
# ШАГ 2: ПРАВИЛЬНАЯ калибровка априора
# ============================================================
# КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: α = λ (без деления на σ²!)
# В Тихонове: min ||Kf-g||² + λ||Df||²
# В байесе:   logL + logP = -0.5*||Kf-g||²/σ² - 0.5*α*||Df||²
# Чтобы MAP байеса = решение Тихонова: α = λ * σ² / σ² = λ
alpha_prior = best_lambda  # Просто λ!
print(f"Коэффициент априора α = {alpha_prior:.3e}")

# Проверка баланса
prior_contrib = alpha_prior * np.sum(np.diff(f_tikhonov, n=2)**2)
like_contrib = np.sum((residual_tikh / noise_std)**2)
print(f"Вклад априора: {prior_contrib:.2f}, вклад правдоподобия: {like_contrib:.2f}")
print(f"Соотношение: {prior_contrib/like_contrib:.2f}")

# ============================================================
# ШАГ 3: Байесовский вывод
# ============================================================
def log_likelihood(theta):
    f = theta[:num_Bloc]
    bg = theta[num_Bloc]
    
    if np.any(f < 0):
        return -np.inf
    
    g_pred = K @ f + bg
    return -0.5 * np.sum(((g_vals - g_pred) / noise_std)**2)

def log_prior(theta):
    f = theta[:num_Bloc]
    
    if np.any(f < 0):
        return -np.inf
    
    if penalty_order == 2:
        d2f = np.diff(f, n=2)
        penalty = -0.5 * alpha_prior * np.sum(d2f**2)
    else:
        df = np.diff(f)
        penalty = -0.5 * alpha_prior * np.sum(df**2)
    
    return penalty

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Инициализация: очень узкий разброс вокруг Тихонова
ndim = num_Bloc + 1
pos = np.zeros((n_walkers, ndim))
for i in range(n_walkers):
    pos[i, :num_Bloc] = np.abs(f_tikhonov + np.random.randn(num_Bloc) * 0.01 * f_tikhonov.max())
    pos[i, num_Bloc] = bg_tikhonov + np.random.randn() * 0.01 * abs(bg_tikhonov)

# Запуск MCMC
sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
print("Запуск MCMC...")
sampler.run_mcmc(pos, n_steps, progress=True)

# Диагностика
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"Autocorrelation time: mean={np.mean(tau):.1f}, max={np.max(tau):.1f}")
    if np.any(tau > n_steps / 50):
        print("⚠ Цепочка может быть недостаточной длины")
    else:
        print("✓ Цепочка сошлась!")
except:
    print("Не удалось оценить autocorrelation time")

# Отбрасываем burn-in
samples = sampler.get_chain(discard=burn_in, flat=True)

# Статистики
f_samples = samples[:, :num_Bloc]
bg_samples = samples[:, num_Bloc]

f_mean = np.mean(f_samples, axis=0)
f_std = np.std(f_samples, axis=0)
bg_mean = np.mean(bg_samples)
bg_std = np.std(bg_samples)

# Невязка
g_rec_bayes = K @ f_mean + bg_mean
residual_bayes = g_vals - g_rec_bayes
norm_residual = np.linalg.norm(residual_bayes)
rel_err = norm_residual / np.linalg.norm(g_vals)

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Фон: {bg_mean:.6f} ± {bg_std:.6f}")
print(f"min(f) = {f_mean.min():.6f}, max(f) = {f_mean.max():.6f}")
print(f"Норма невязки: {norm_residual:.6f}")
print(f"Относительная ошибка: {rel_err*100:.2f}%")
print(f"Сравнение с Тихоновым: {np.linalg.norm(residual_tikh)/np.linalg.norm(g_vals)*100:.2f}%")

# Сохранение
f_output_filename = os.path.join(output_dir, f"{temp_label}_f_Bloc_bayesian.txt")
f_data = np.column_stack((Bloc_vals, f_mean, f_std))
np.savetxt(f_output_filename, f_data, fmt='%.8e', delimiter='\t',
           header=f"B_loc(T)\tf(B_loc)\terror\nFile: {os.path.basename(filename)}\nMethod: Bayesian MCMC\nlambda_equiv={best_lambda:.3e}\nnoise_std={noise_std:.3e}\nrel_err={rel_err:.4f}",
           comments='')

print(f"Сохранено: {f_output_filename}")

# Визуализация
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(Bloc_vals, f_mean, 'b-', label='Bayesian mean', lw=2)
plt.fill_between(Bloc_vals, f_mean - f_std, f_mean + f_std, alpha=0.3, color='blue', label='±1σ')
plt.plot(Bloc_vals, f_tikhonov, 'r--', label='Tikhonov', alpha=0.7)
plt.title('Recovery Comparison')
plt.xlabel('B_loc (T)')
plt.ylabel('f(B_loc)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(B_vals, g_vals, 'k.', label='data', markersize=3)
plt.plot(B_vals, g_rec_bayes, 'b-', label='Bayesian', lw=2)
plt.plot(B_vals, g_rec_tikh, 'r--', label='Tikhonov', alpha=0.7)
plt.legend()
plt.xlabel('B (T)')
plt.ylabel('Intensity')
plt.title(f'Rel. error: Bayes={rel_err*100:.2f}%, Tikh={np.linalg.norm(residual_tikh)/np.linalg.norm(g_vals)*100:.2f}%')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{temp_label}_bayesian.png"), dpi=150)
plt.show()