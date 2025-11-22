import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import lsq_linear


filename = r"C:\Users\Владимир\Desktop\Diplom\my_distribution_gB.txt"
BL = 5
Bloc_min = 0.01
Bloc_max = 3
num_Bloc = 100
use_nonneg = True
penalty_order = 2
auto_lambda = True
lambda_range = np.logspace(0, 5, 100)

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


idx = np.argsort(B_vals)
B_vals, g_vals = B_vals[idx], g_vals[idx]

def K_new(B, Bloc, BL):
    if B == 0 or Bloc == 0:
        return 0.0
    if Bloc < abs(B - BL):
        return 0.0
    return (B**2 - Bloc**2 + BL**2) / (Bloc * B**2)

def build_matrices(B_vals, BL, Bloc_vals, penalty_order):
    K = np.zeros((len(B_vals), len(Bloc_vals)))
    for i, B in enumerate(B_vals):
        for j, Bloc in enumerate(Bloc_vals):
            K[i, j] = K_new(B, Bloc, BL)
    K_ext = np.hstack([K, np.ones((len(B_vals), 1))])
    
    num_Bloc = len(Bloc_vals)
    if penalty_order == 2:
        D = (np.diag(np.ones(num_Bloc-1), -1)
             - 2*np.diag(np.ones(num_Bloc), 0)
             + np.diag(np.ones(num_Bloc-1), 1))
    elif penalty_order == 1:
        D = np.diff(np.eye(num_Bloc), axis=0)
    else:
        raise ValueError("penalty_order = 1 or 2")
    
    D_ext = np.zeros((D.shape[0], num_Bloc + 1))
    D_ext[:, :num_Bloc] = D
    return K, K_ext, D_ext


def solve_with_lambda(B_vals, g_vals, K_ext, D_ext, lambda_val, use_nonneg):
    sqrt_lambda = np.sqrt(lambda_val)
    A_aug = np.vstack([K_ext, sqrt_lambda * D_ext])
    b_aug = np.hstack([g_vals, np.zeros(D_ext.shape[0])])
    
    n = K_ext.shape[1]  
    if use_nonneg:
        lb = np.zeros(n)
        ub = np.full(n, np.inf)
        lb[-1] = -1e20  
    else:
        lb = np.full(n, -1e20)
        ub = np.full(n, 1e20)
    
    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), lsmr_tol='auto')
    return res.x, res.cost


def compute_gcv_for_part(B_vals, g_vals, K, K_ext, D_ext, lambda_val):
    sol, _ = solve_with_lambda(B_vals, g_vals, K_ext, D_ext, lambda_val, use_nonneg)
    g_pred = K_ext @ sol
    residual = g_vals - g_pred
    try:
        U, s, Vt = np.linalg.svd(K, full_matrices=False)
        trace_H = np.sum(s**2 / (s**2 + lambda_val))
    except np.linalg.LinAlgError:
        trace_H = len(B_vals) * 0.5 
    N = len(g_vals)
    denom = (N - trace_H)
    if denom <= 0:
        return np.inf
    gcv = np.linalg.norm(residual)**2 / (denom**2)
    return gcv


def find_lambda_gcv_part(B_vals, g_vals, K, K_ext, D_ext, lambda_range):
    gcv_vals = []
    for lam in lambda_range:
        try:
            gcv = compute_gcv_for_part(B_vals, g_vals, K, K_ext, D_ext, lam)
            gcv_vals.append(gcv)
        except:
            gcv_vals.append(np.inf)
    gcv_vals = np.array(gcv_vals)
    idx_opt = np.argmin(gcv_vals)
    return lambda_range[idx_opt], gcv_vals, idx_opt


def reconstruct_part(name, B_vals, g_vals, BL, Bloc_vals):
    print(f"\n🔄 Восстановление: {name}")
    K, K_ext, D_ext = build_matrices(B_vals, BL, Bloc_vals, penalty_order)
    
    if auto_lambda:
        lambda_opt, gcv_vals, idx_opt = find_lambda_gcv_part(B_vals, g_vals, K, K_ext, D_ext, lambda_range)
        print(f"  λ_opt ({name}) = {lambda_opt:.3e}")
    else:
        lambda_opt = 1.0 
    
    sol, _ = solve_with_lambda(B_vals, g_vals, K_ext, D_ext, lambda_opt, use_nonneg)
    f_sol = sol[:-1]
    bg = sol[-1]
    g_rec = K_ext @ sol
    
    return {
        'name': name,
        'f': f_sol,
        'bg': bg,
        'lambda': lambda_opt,
        'K': K,
        'g_rec': g_rec,
        'B_vals': B_vals,
        'g_vals': g_vals
    }


Bloc_vals = np.linspace(Bloc_min, Bloc_max, num_Bloc)

mask_left = B_vals <= BL
mask_right = B_vals >= BL

B_left, g_left = B_vals[mask_left], g_vals[mask_left]
B_right, g_right = B_vals[mask_right], g_vals[mask_right]

res_full = reconstruct_part("full", B_vals, g_vals, BL, Bloc_vals)
res_left = reconstruct_part("left", B_left, g_left, BL, Bloc_vals)
res_right = reconstruct_part("right", B_right, g_right, BL, Bloc_vals)

def reconstruct_on_full_B(res_part, B_full, BL, Bloc_vals):
    K_full = np.zeros((len(B_full), len(Bloc_vals)))
    for i, B in enumerate(B_full):
        for j, Bloc in enumerate(Bloc_vals):
            K_full[i, j] = K_new(B, Bloc, BL)
    return K_full @ res_part['f'] + res_part['bg']


g_rec_left_on_full = reconstruct_on_full_B(res_left, B_vals, BL, Bloc_vals)
g_rec_right_on_full = reconstruct_on_full_B(res_right, B_vals, BL, Bloc_vals)

def normalize(f, x):
    integ = np.trapz(f, x)
    return f / integ if integ > 0 else f


f_full_norm = normalize(res_full['f'], Bloc_vals)
f_left_norm = normalize(res_left['f'], Bloc_vals)
f_right_norm = normalize(res_right['f'], Bloc_vals)

plt.figure(figsize=(15, 5))

# 1. Распределения
plt.subplot(1, 3, 1)
plt.plot(Bloc_vals, f_full_norm, 'k-', label='Полный', linewidth=2)
plt.plot(Bloc_vals, f_left_norm, 'b--', label='Левая', linewidth=2)
plt.plot(Bloc_vals, f_right_norm, 'r-.', label='Правая', linewidth=2)
plt.xlabel(r'$B_{\mathrm{loc}}$ ')
plt.ylabel(r'$f(B_{\mathrm{loc}})$ (норм.)')
plt.title('Распределения локальных полей')
plt.grid(True, alpha=0.5)
plt.legend()

# 2. Сигналы
plt.subplot(1, 3, 2)
plt.plot(B_vals, g_vals, 'k-', label='Эксперимент', alpha=0.8)
plt.plot(B_vals, res_full['g_rec'], 'g--', label='Восст. (полный)', alpha=0.9)
plt.plot(B_vals, g_rec_left_on_full, 'b:', label='Восст. (из левой)', alpha=0.8)
plt.plot(B_vals, g_rec_right_on_full, 'r:', label='Восст. (из правой)', alpha=0.8)
plt.axvline(BL, color='gray', linestyle=':', label=r'$B_L$')
plt.xlabel(r'$B$')
plt.ylabel(r'$g(B)$')
plt.title('Восстановленные сигналы')
plt.grid(True, alpha=0.5)
plt.legend(fontsize=9)

# 3. Разность профилей
plt.subplot(1, 3, 3)
diff = f_left_norm - f_right_norm
plt.plot(Bloc_vals, diff, 'm-', linewidth=2)
plt.fill_between(Bloc_vals, diff, 0, color='magenta', alpha=0.3)
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel(r'$B_{\mathrm{loc}}$')
plt.ylabel(r'$f_{\mathrm{left}} - f_{\mathrm{right}}$')
plt.title('Разность нормированных профилей')
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ:")
print(f"Фон (полный)  : {res_full['bg']:.6e}")
print(f"Фон (левая)   : {res_left['bg']:.6e}")
print(f"Фон (правая)  : {res_right['bg']:.6e}")
print(f"Δ фон (прав-лев): {res_right['bg'] - res_left['bg']:.2e}")

print(f"\nλ (полный)    : {res_full['lambda']:.3e}")
print(f"λ (левая)     : {res_left['lambda']:.3e}")
print(f"λ (правая)    : {res_right['lambda']:.3e}")

def rel_err(true, rec):
    return np.linalg.norm(true - rec) / np.linalg.norm(true)

print("\nОтносительные ошибки восстановления на полной сетке B:")
print(f"  Полный   : {rel_err(g_vals, res_full['g_rec']):.2%}")
print(f"  Из левой : {rel_err(g_vals, g_rec_left_on_full):.2%}")
print(f"  Из правой: {rel_err(g_vals, g_rec_right_on_full):.2%}")

print(f"\nНорма разности f_left - f_right: {np.linalg.norm(f_left_norm - f_right_norm):.3f}")
print("\nАнализ завершен")
