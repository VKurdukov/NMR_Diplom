import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

# =============================================================================
# НАСТРОЙКИ ШРИФТОВ
# =============================================================================
FONT_SCALE = 1.5

plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams.update({
    'font.size': 10 * FONT_SCALE,
    'axes.titlesize': 12 * FONT_SCALE,
    'axes.labelsize': 11 * FONT_SCALE,
    'xtick.labelsize': 10 * FONT_SCALE,
    'ytick.labelsize': 10 * FONT_SCALE,
    'legend.fontsize': 10 * FONT_SCALE,
})

# Начальные приближения
T_N2_init = 9.95
T_N1_init = 8.17

# Максимальная температура для анализа
T_MAX = 15.0

folder_path = r"C:\Users\Владимир\Desktop\NMR_Diplom\bootstrap_results"

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

temps = []
centroid_list = []
centroid_err_list = []
std_max_list = []
std_max_err_list = []

for filename in txt_files:
    filepath = os.path.join(folder_path, filename)
    
    match = re.search(r'(\d+\.\d+)K', filename)
    if not match:
        continue
    temp = float(match.group(1))
    
    if temp > T_MAX:
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        continue

    x_vals, y_vals, y_err = [], [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", parts[0]):
                try:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                    y_err.append(float(parts[2]))
                except ValueError:
                    continue

    if len(x_vals) == 0:
        continue

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    y_err = np.array(y_err)

    y_max = np.max(y_vals)
    if y_max == 0:
        continue
    
    y_norm = y_vals / y_max
    y_err_norm = y_err / y_max

    sum_y = np.sum(y_norm)
    centroid = np.sum(x_vals * y_norm) / sum_y
    sigma_centroid = np.sqrt(np.sum((x_vals - centroid)**2 * y_err_norm**2)) / sum_y

    max_idx = np.argmax(y_norm)
    x_max = x_vals[max_idx]

    S = np.sum(y_norm * (x_vals - x_max)**2)
    std_max = np.sqrt(S / sum_y)

    if std_max > 1e-10:
        d_std_dy = ((x_vals - x_max)**2 - std_max**2) / (2 * std_max * sum_y)
        sigma_std = np.sqrt(np.sum(d_std_dy**2 * y_err_norm**2))
    else:
        sigma_std = 0.0

    temps.append(temp)
    centroid_list.append(centroid)
    centroid_err_list.append(sigma_centroid)
    std_max_list.append(std_max)
    std_max_err_list.append(sigma_std)

# Сортировка по температуре
temps = np.array(temps)
centroid_list = np.array(centroid_list)
centroid_err_list = np.array(centroid_err_list)
std_max_list = np.array(std_max_list)
std_max_err_list = np.array(std_max_err_list)

sort_idx = np.argsort(temps)
temps = temps[sort_idx]
centroid_list = centroid_list[sort_idx]
centroid_err_list = centroid_err_list[sort_idx]
std_max_list = std_max_list[sort_idx]
std_max_err_list = std_max_err_list[sort_idx]

print(f"📊 Обработано точек: {len(temps)} (T ≤ {T_MAX} K)")
print(f"   Диапазон температур: {temps.min():.2f} – {temps.max():.2f} K")

# =============================================================================
# МОДЕЛЬ 1: КУСОЧНО-ЛИНЕЙНАЯ (для σ_B)
# =============================================================================
def piecewise_model(T, params, T_N1, T_N2):
    """Кусочно-линейная функция. params = [y1, s1, s2, s3]"""
    y1, s1, s2, s3 = params
    T = np.atleast_1d(T)
    y = np.zeros_like(T, dtype=float)
    
    mask1 = T < T_N1
    mask2 = (T >= T_N1) & (T < T_N2)
    mask3 = T >= T_N2
    
    y[mask1] = y1 + s1 * (T[mask1] - T_N1)
    y[mask2] = y1 + s2 * (T[mask2] - T_N1)
    y[mask3] = (y1 + s2 * (T_N2 - T_N1)) + s3 * (T[mask3] - T_N2)
    
    return y

def build_design_matrix(T, T_N1, T_N2):
    """Матрица A для кусочно-линейной модели."""
    T = np.atleast_1d(T)
    n = len(T)
    A = np.zeros((n, 4))
    
    mask1 = T < T_N1
    mask2 = (T >= T_N1) & (T < T_N2)
    mask3 = T >= T_N2
    
    A[mask1, 0] = 1.0
    A[mask1, 1] = T[mask1] - T_N1
    
    A[mask2, 0] = 1.0
    A[mask2, 2] = T[mask2] - T_N1
    
    A[mask3, 0] = 1.0
    A[mask3, 2] = T_N2 - T_N1
    A[mask3, 3] = T[mask3] - T_N2
    
    return A

def fit_piecewise(T, y, y_err, T_N1, T_N2):
    """Взвешенный МНК для кусочно-линейной модели."""
    mask = y_err > 0
    T_fit = T[mask]
    y_fit = y[mask]
    y_err_fit = y_err[mask]
    
    if len(T_fit) < 4:
        return None, None, np.inf, 0
    
    A = build_design_matrix(T_fit, T_N1, T_N2)
    W = np.diag(1.0 / y_err_fit**2)
    
    ATA = A.T @ W @ A
    ATy = A.T @ W @ y_fit
    
    try:
        params = np.linalg.solve(ATA, ATy)
        cov = np.linalg.inv(ATA)
    except np.linalg.LinAlgError:
        return None, None, np.inf, 0
    
    params_err = np.sqrt(np.abs(np.diag(cov)))
    y_pred = A @ params
    chi2 = np.sum(((y_fit - y_pred) / y_err_fit)**2)
    dof = len(y_fit) - 4
    
    return params, params_err, chi2, dof

def optimize_transitions(T, y, y_err, T_N1_init, T_N2_init):
    """Оптимизация T_N1 и T_N2 для кусочно-линейной модели."""
    def objective(bp):
        T_N1, T_N2 = bp
        if T_N1 >= T_N2:
            return 1e10
        if T_N1 < T.min() or T_N2 > T.max():
            return 1e10
        n1 = np.sum(T < T_N1)
        n2 = np.sum((T >= T_N1) & (T < T_N2))
        n3 = np.sum(T >= T_N2)
        if n1 < 2 or n2 < 2 or n3 < 2:
            return 1e10
        
        params, _, chi2, dof = fit_piecewise(T, y, y_err, T_N1, T_N2)
        if params is None or dof <= 0:
            return 1e10
        return chi2 / dof
    
    x0 = [T_N1_init, T_N2_init]
    bounds = [(T.min() + 0.5, T_N2_init - 0.3),
              (T_N1_init + 0.3, T.max() - 0.5)]
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 1000})
    
    T_N1_opt, T_N2_opt = result.x
    params, params_err, chi2, dof = fit_piecewise(T, y, y_err, T_N1_opt, T_N2_opt)
    
    return T_N1_opt, T_N2_opt, params, params_err, chi2, dof

# =============================================================================
# МОДЕЛЬ 2: ПРЯМАЯ + СТЕПЕННАЯ + ПРЯМАЯ (для ⟨B_loc⟩)
# =============================================================================
# Параметризация через 5 свободных параметров: [y1, s1, s3, T_c, beta]
# y2 вычисляется из условия непрерывности: y2 = y1 * ((T_c - T_N2)/(T_c - T_N1))^beta
# 
# Модель:
#   T < T_N1:     y = y1 + s1*(T - T_N1)
#   T_N1 ≤ T < T_N2: y = y1 * ((T_c - T)/(T_c - T_N1))^beta
#   T ≥ T_N2:     y = y2 + s3*(T - T_N2)
# где y2 = y1 * ((T_c - T_N2)/(T_c - T_N1))^beta

def composite_model(T, params, T_N1, T_N2):
    """
    Составная модель: прямая + степенная + прямая.
    params = [y1, s1, s3, T_c, beta]
    Гарантирует непрерывность в T_N1 и T_N2.
    """
    y1, s1, s3, T_c, beta = params
    T = np.atleast_1d(T)
    y = np.zeros_like(T, dtype=float)
    
    # Значение в T_N2 из условия непрерывности
    ratio = (T_c - T_N2) / (T_c - T_N1)
    if ratio <= 0:
        return np.full_like(T, np.nan)
    y2 = y1 * ratio**beta
    
    mask1 = T < T_N1
    mask2 = (T >= T_N1) & (T < T_N2)
    mask3 = T >= T_N2
    
    # Левая прямая
    y[mask1] = y1 + s1 * (T[mask1] - T_N1)
    
    # Степенная (центр)
    # Защита от отрицательных значений
    x_center = np.clip((T_c - T[mask2]) / (T_c - T_N1), 1e-10, None)
    y[mask2] = y1 * x_center**beta
    
    # Правая прямая
    y[mask3] = y2 + s3 * (T[mask3] - T_N2)
    
    return y

def fit_composite(T, y, y_err, T_N1, T_N2, p0=None):
    """
    Нелинейный взвешенный МНК для составной модели.
    """
    mask = y_err > 0
    T_fit = T[mask]
    y_fit = y[mask]
    y_err_fit = y_err[mask]
    
    if len(T_fit) < 5:
        return None, None, np.inf, 0
    
    # Начальные приближения
    if p0 is None:
        y1_guess = y_fit.max() * 1.2
        s1_guess = -0.01
        s3_guess = -0.001
        T_c_guess = T_N2 * 1.05
        beta_guess = 0.35
        p0 = [y1_guess, s1_guess, s3_guess, T_c_guess, beta_guess]
    
    def model_func(T_arr, y1, s1, s3, T_c, beta):
        return composite_model(T_arr, [y1, s1, s3, T_c, beta], T_N1, T_N2)
    
    # Ограничения
    bounds = (
        [0, -1.0, -1.0, T_N2 * 0.9, 0.1],      # нижние
        [1.0, 0.0, 1.0, T_N2 * 1.5, 10.0]       # верхние
    )
    
    try:
        popt, pcov = curve_fit(
            model_func, T_fit, y_fit,
            p0=p0,
            sigma=y_err_fit,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=20000
        )
        
        perr = np.sqrt(np.abs(np.diag(pcov)))
        
        y_pred = model_func(T_fit, *popt)
        chi2 = np.sum(((y_fit - y_pred) / y_err_fit)**2)
        dof = len(y_fit) - 5
        
        return popt, perr, chi2, dof
    except Exception as e:
        print(f"   ⚠️ Ошибка фитирования: {e}")
        return None, None, np.inf, 0

def optimize_composite(T, y, y_err, T_N1_init, T_N2_init):
    """
    Оптимизация T_N1, T_N2 и параметров составной модели.
    """
    def objective(params):
        T_N1, T_N2 = params[0], params[1]
        if T_N1 >= T_N2:
            return 1e10
        if T_N1 < T.min() or T_N2 > T.max():
            return 1e10
        
        # Проверяем, что в каждом сегменте достаточно точек
        n1 = np.sum(T < T_N1)
        n2 = np.sum((T >= T_N1) & (T < T_N2))
        n3 = np.sum(T >= T_N2)
        if n1 < 2 or n2 < 2 or n3 < 2:
            return 1e10
        
        p0_rest = params[2:]
        result, _, chi2, dof = fit_composite(T, y, y_err, T_N1, T_N2, p0=p0_rest)
        if result is None or dof <= 0:
            return 1e10
        return chi2 / dof
    
    # Начальные приближения: [T_N1, T_N2, y1, s1, s3, T_c, beta]
    y_max = y.max()
    x0 = [T_N1_init, T_N2_init, 
          y_max * 1.2, -0.01, -0.001, T_N2_init * 1.05, 0.35]
    
    bounds = [
        (T.min() + 0.5, T_N2_init - 0.3),      # T_N1
        (T_N1_init + 0.3, T.max() - 0.5),      # T_N2
        (0, 1.0),                                # y1
        (-1.0, 0.0),                             # s1
        (-1.0, 1.0),                             # s3
        (T_N2_init * 0.9, T_N2_init * 1.5),    # T_c
        (0.1, 10.0)                               # beta
    ]
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-6, 
                               'maxiter': 5000, 'adaptive': True})
    
    T_N1_opt, T_N2_opt = result.x[0], result.x[1]
    p0_final = result.x[2:]
    
    # Финальное фитирование с оптимальными температурами
    params, params_err, chi2, dof = fit_composite(
        T, y, y_err, T_N1_opt, T_N2_opt, p0=p0_final
    )
    
    return T_N1_opt, T_N2_opt, params, params_err, chi2, dof

# =============================================================================
# АППРОКСИМАЦИЯ ШИРИНЫ РАСПРЕДЕЛЕНИЯ (кусочно-линейная)
# =============================================================================
print("\n" + "=" * 70)
print("АППРОКСИМАЦИЯ σ_B(T) — кусочно-линейная модель")
print("=" * 70)

T_N1_opt_std, T_N2_opt_std, params_std, params_err_std, chi2_std, dof_std = \
    optimize_transitions(temps, std_max_list, std_max_err_list, T_N1_init, T_N2_init)

if params_std is not None:
    y1, s1, s2, s3 = params_std
    y1_err, s1_err, s2_err, s3_err = params_err_std
    y2 = y1 + s2 * (T_N2_opt_std - T_N1_opt_std)
    y2_err = np.sqrt(y1_err**2 + (T_N2_opt_std - T_N1_opt_std)**2 * s2_err**2)
    
    print(f"\nОПТИМАЛЬНЫЕ ТЕМПЕРАТУРЫ:")
    print(f"  T_N1 = {T_N1_opt_std:.3f} К")
    print(f"  T_N2 = {T_N2_opt_std:.3f} К")
    print(f"\nПАРАМЕТРЫ:")
    print(f"  Значение в T_N1: y₁ = {y1:.5f} ± {y1_err:.5f} Тл")
    print(f"  Значение в T_N2: y₂ = {y2:.5f} ± {y2_err:.5f} Тл")
    print(f"  Наклон AFM (T < T_N1):           s₁ = {s1:.5f} ± {s1_err:.5f} Тл/К")
    print(f"  Наклон пром. фаза (T_N1–T_N2):   s₂ = {s2:.5f} ± {s2_err:.5f} Тл/К")
    print(f"  Наклон парамагн. (T > T_N2):     s₃ = {s3:.5f} ± {s3_err:.5f} Тл/К")
    print(f"  χ²/dof = {chi2_std:.2f} / {dof_std} = {chi2_std/dof_std:.3f}")

# =============================================================================
# АППРОКСИМАЦИЯ СРЕДНЕГО ПОЛЯ (прямая + степенная + прямая)
# =============================================================================
print("\n" + "=" * 70)
print("АППРОКСИМАЦИЯ ⟨B_loc⟩(T) — прямая + степенная + прямая")
print("=" * 70)

T_N1_opt_mean, T_N2_opt_mean, params_mean, params_err_mean, chi2_mean, dof_mean = \
    optimize_composite(temps, centroid_list, centroid_err_list, T_N1_init, T_N2_init)

if params_mean is not None:
    y1, s1, s3, T_c, beta = params_mean
    y1_err, s1_err, s3_err, T_c_err, beta_err = params_err_mean
    
    # Значение в T_N2 из непрерывности
    ratio = (T_c - T_N2_opt_mean) / (T_c - T_N1_opt_mean)
    y2 = y1 * ratio**beta
    # Погрешность y2 через propagation
    dy2_dy1 = ratio**beta
    dy2_dTc = y1 * beta * ratio**(beta-1) * (-(T_N2_opt_mean - T_N1_opt_mean)/(T_c - T_N1_opt_mean)**2)
    dy2_dbeta = y1 * ratio**beta * np.log(ratio)
    y2_err = np.sqrt((dy2_dy1*y1_err)**2 + (dy2_dTc*T_c_err)**2 + (dy2_dbeta*beta_err)**2)
    
    print(f"\nОПТИМАЛЬНЫЕ ТЕМПЕРАТУРЫ:")
    print(f"  T_N1 = {T_N1_opt_mean:.3f} К")
    print(f"  T_N2 = {T_N2_opt_mean:.3f} К")
    print(f"\nПАРАМЕТРЫ МОДЕЛИ:")
    print(f"  Значение в T_N1: y₁ = {y1:.5f} ± {y1_err:.5f} Тл")
    print(f"  Значение в T_N2: y₂ = {y2:.5f} ± {y2_err:.5f} Тл")
    print(f"\nЛЕВАЯ ПРЯМАЯ (T < T_N1):")
    print(f"  y = y₁ + s₁·(T - T_N1)")
    print(f"  s₁ = {s1:.5f} ± {s1_err:.5f} Тл/К")
    print(f"\nСТЕПЕННАЯ (T_N1 ≤ T < T_N2):")
    print(f"  y = y₁ · ((T_c - T)/(T_c - T_N1))^β")
    print(f"  b₀ = y₁ = {y1:.5f} ± {y1_err:.5f} Тл")
    print(f"  T_c = {T_c:.3f} ± {T_c_err:.3f} К")
    print(f"  β = {beta:.3f} ± {beta_err:.3f}")
    print(f"\nПРАВАЯ ПРЯМАЯ (T ≥ T_N2):")
    print(f"  y = y₂ + s₃·(T - T_N2)")
    print(f"  s₃ = {s3:.5f} ± {s3_err:.5f} Тл/К")
    print(f"\nКАЧЕСТВО:")
    print(f"  χ²/dof = {chi2_mean:.2f} / {dof_mean} = {chi2_mean/dof_mean:.3f}")
    
    # Интерпретация критического индекса
    print(f"\nИНТЕРПРЕТАЦИЯ β:")
    if abs(beta - 0.32) < 0.1:
        print(f"  β ≈ 0.32 → модель 3D Изинга")
    elif abs(beta - 0.36) < 0.1:
        print(f"  β ≈ 0.36 → модель 3D Гейзенберга")
    elif abs(beta - 0.5) < 0.1:
        print(f"  β ≈ 0.50 → среднее поле")
    elif abs(beta - 0.125) < 0.1:
        print(f"  β ≈ 0.125 → модель 2D Изинга")
    else:
        print(f"  β = {beta:.3f} → нестандартный критический индекс")

# =============================================================================
# ГРАФИКИ
# =============================================================================
xticks = np.arange(np.floor(temps.min()), np.ceil(temps.max()) + 1, 1)
xtick_labels = ['%d' % int(t) for t in xticks]

# --- График 1: Среднее локальное поле (составная модель) ---
fig1, ax1 = plt.subplots(figsize=(10, 7))

ax1.errorbar(temps, centroid_list, yerr=centroid_err_list, 
             fmt='o', color='tab:blue', markersize=7, 
             capsize=4, capthick=1.5, elinewidth=1.5, zorder=3,
             label=r'Эксперимент: $\langle B_{\mathrm{loc}} \rangle$')

if params_mean is not None:
    # Построение гладкой кривой
    t_fine = np.linspace(temps.min() - 0.5, T_MAX, 1000)
    y_fine = composite_model(t_fine, params_mean, T_N1_opt_mean, T_N2_opt_mean)
    
    # Раскрашиваем участки разными цветами
    mask_left = t_fine < T_N1_opt_mean
    mask_center = (t_fine >= T_N1_opt_mean) & (t_fine < T_N2_opt_mean)
    mask_right = t_fine >= T_N2_opt_mean
    
    ax1.plot(t_fine[mask_left], y_fine[mask_left], '-', color='green', 
             linewidth=2.5, zorder=2, label=f'Прямая (AFM)')
    ax1.plot(t_fine[mask_center], y_fine[mask_center], '-', color='darkorange', 
             linewidth=2.5, zorder=2, 
             label=rf'Степенная: $b_0(1-T/T_c)^\beta$, $\beta$={beta:.2f}')
    ax1.plot(t_fine[mask_right], y_fine[mask_right], '-', color='gray', 
             linewidth=2.5, zorder=2, label='Прямая (парамагн.)')
    
    # Точки излома
    y_at_TN1 = params_mean[0]
    ratio = (params_mean[3] - T_N2_opt_mean) / (params_mean[3] - T_N1_opt_mean)
    y_at_TN2 = params_mean[0] * ratio**params_mean[4]
    ax1.plot([T_N1_opt_mean, T_N2_opt_mean], [y_at_TN1, y_at_TN2], 'D', color='red', 
             markersize=10, zorder=4, label='Точки излома')

ax1.axvline(x=T_N1_opt_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.axvline(x=T_N2_opt_mean, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)

ax1.text(T_N1_opt_mean, 0.6, f'$T_{{N1}}={T_N1_opt_mean:.1f}$ К', color='red', 
         fontsize=10 * FONT_SCALE, ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))
ax1.text(T_N2_opt_mean, 0.6, f'$T_{{N2}}={T_N2_opt_mean:.1f}$ К', color='purple', 
         fontsize=10 * FONT_SCALE, ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='purple', alpha=0.8))

ax1.set_xlabel('Температура $T$, К')
ax1.set_ylabel(r'Среднее локальное поле $\langle B_{\mathrm{loc}} \rangle$, Тл')
ax1.set_xlim(temps.min() - 0.5, T_MAX)
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)
fig1.tight_layout()
plt.savefig('fig_mean_composite.png', dpi=300, bbox_inches='tight')
plt.show()

# --- График 2: Ширина распределения (кусочно-линейная) ---
fig2, ax2 = plt.subplots(figsize=(10, 7))

ax2.errorbar(temps, std_max_list, yerr=std_max_err_list, 
             fmt='s', color='tab:red', markersize=7, 
             capsize=4, capthick=1.5, elinewidth=1.5, zorder=3,
             label=r'Эксперимент: $\sigma_B$')

if params_std is not None:
    t_fine = np.linspace(temps.min() - 0.5, T_MAX, 500)
    y_fine = piecewise_model(t_fine, params_std, T_N1_opt_std, T_N2_opt_std)
    
    # Раскрашиваем участки
    mask_left = t_fine < T_N1_opt_std
    mask_center = (t_fine >= T_N1_opt_std) & (t_fine < T_N2_opt_std)
    mask_right = t_fine >= T_N2_opt_std
    
    ax2.plot(t_fine[mask_left], y_fine[mask_left], '-', color='green', 
             linewidth=2.5, zorder=2, label='Прямая (AFM)')
    ax2.plot(t_fine[mask_center], y_fine[mask_center], '-', color='darkorange', 
             linewidth=2.5, zorder=2, label='Прямая (пром. фаза)')
    ax2.plot(t_fine[mask_right], y_fine[mask_right], '-', color='gray', 
             linewidth=2.5, zorder=2, label='Прямая (парамагн.)')
    
    y_at_TN1 = params_std[0]
    y_at_TN2 = params_std[0] + params_std[2] * (T_N2_opt_std - T_N1_opt_std)
    ax2.plot([T_N1_opt_std, T_N2_opt_std], [y_at_TN1, y_at_TN2], 'D', color='blue', 
             markersize=10, zorder=4, label='Точки излома')

ax2.axvline(x=T_N1_opt_std, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.axvline(x=T_N2_opt_std, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)

ax2.text(T_N1_opt_std, 0.65, f'$T_{{N1}}={T_N1_opt_std:.1f}$ К', color='red', 
         fontsize=10 * FONT_SCALE, ha='center', va='bottom', transform=ax2.get_xaxis_transform(),
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))
ax2.text(T_N2_opt_std, 0.65, f'$T_{{N2}}={T_N2_opt_std:.1f}$ К', color='purple', 
         fontsize=10 * FONT_SCALE, ha='center', va='bottom', transform=ax2.get_xaxis_transform(),
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='purple', alpha=0.8))

ax2.set_xlabel('Температура $T$, К')
ax2.set_ylabel(r'Ширина распределения $\sigma_B$, Тл')
ax2.set_xlim(temps.min() - 0.5, T_MAX)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xtick_labels)
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.5)
fig2.tight_layout()
plt.savefig('fig_std_piecewise.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# ГРАФИК 3: ОТНОШЕНИЕ σ / ⟨B⟩
# =============================================================================
valid = np.abs(centroid_list) > 1e-4
t_ratio = temps[valid]
ratio = std_max_list[valid] / centroid_list[valid]

ratio_err = np.zeros_like(ratio)
for i, idx in enumerate(np.where(valid)[0]):
    mu = centroid_list[idx]
    sig = std_max_list[idx]
    mu_err = centroid_err_list[idx]
    sig_err = std_max_err_list[idx]
    R = sig / mu
    ratio_err[i] = abs(R) * np.sqrt((sig_err / sig)**2 + (mu_err / mu)**2)

fig3, ax3 = plt.subplots(figsize=(10, 7))

ax3.errorbar(t_ratio, ratio, yerr=ratio_err, 
             fmt='D-', color='tab:green', markersize=7, 
             capsize=4, capthick=1.5, elinewidth=1.5,
             label=r'$\sigma_B / \langle B_{\mathrm{loc}} \rangle$')

T_N1_avg = 0.5 * (T_N1_opt_mean + T_N1_opt_std)
T_N2_avg = 0.5 * (T_N2_opt_mean + T_N2_opt_std)

ax3.axvline(x=T_N1_avg, color='red', linestyle='--', linewidth=1.5)
ax3.axvline(x=T_N2_avg, color='purple', linestyle='--', linewidth=1.5)

ax3.text(T_N1_avg, 0.95, f'$T_{{N1}}={T_N1_avg:.1f}$ К', color='red', 
         fontsize=10 * FONT_SCALE, ha='right', va='top', transform=ax3.get_xaxis_transform(),
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))
ax3.text(T_N2_avg, 0.95, f'$T_{{N2}}={T_N2_avg:.1f}$ К', color='purple', 
         fontsize=10 * FONT_SCALE, ha='left', va='top', transform=ax3.get_xaxis_transform(),
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='purple', alpha=0.8))

ax3.set_xlabel('Температура $T$, К')
ax3.set_ylabel(r'Отношение $\sigma_B / \langle B_{\mathrm{loc}} \rangle$')
ax3.set_xlim(temps.min() - 0.5, T_MAX)
ax3.set_xticks(xticks)
ax3.set_xticklabels(xtick_labels)
ax3.legend(loc='best', framealpha=0.9)
ax3.grid(True, linestyle='--', alpha=0.5)
fig3.tight_layout()
plt.savefig('fig_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Готово! Сохранено:")
print("   - fig_mean_composite.png")
print("   - fig_std_piecewise.png")
print("   - fig_ratio.png")
print(f"Все графики обрезаны до T ≤ {T_MAX} K")
print(f"Обработано файлов: {len(temps)}")