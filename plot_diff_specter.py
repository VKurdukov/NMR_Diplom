import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter

# Папка с файлами вида *_f_Bloc.txt
folder = Path("final_data")

spectra = []

# === ЗАГРУЗКА ВСЕХ СПЕКТРОВ ===
for file in folder.glob("*_f_Bloc.txt"):
    name = file.stem  # например: "3.50K_f_Bloc"
    temp_str = name.replace("_f_Bloc", "")
    match = re.search(r"(\d+\.?\d*)", temp_str)
    if not match:
        print(f"Не удалось извлечь температуру из {file.name}")
        continue
    temp = float(match.group(1))

    try:
        data = np.loadtxt(file, skiprows=3)
        B_loc = data[:, 0]
        f = data[:, 1]

        integral = np.trapezoid(f, B_loc)
        if integral <= 0:
            print(f"Пропущен {file.name}: интеграл = {integral:.2e} ≤ 0")
            continue

        f_norm = f / integral
        spectra.append((temp, B_loc, f_norm))
    except Exception as e:
        print(f"Ошибка при загрузке {file.name}: {e}")
        continue

if not spectra:
    raise RuntimeError("Нет корректных данных f(B_loc) в папке 'final_gauss_data'")

# Сортируем по температуре
spectra.sort(key=lambda x: x[0])
temps = [t for t, _, _ in spectra]

# Цветовая палитра: синий (низкие T) → красный (высокие T)
cmap = LinearSegmentedColormap.from_list("blue_red", ["#00446C", "#FF0000"])
norm = Normalize(vmin=min(temps), vmax=max(temps))
colors = [cmap(norm(t)) for t in temps]

# === ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ДЛЯ ВСЕХ СПЕКТРОВ ===
derivatives_1st = []  # (temp, B_loc, df/dB)
derivatives_2nd = []  # (temp, B_loc, d²f/dB²)

for temp, B_loc, f_norm in spectra:
    # Сглаживание перед дифференцированием (фильтр Савицкого-Голея)
    window_length = min(11, len(f_norm) // 2 * 2 + 1)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1
    
    f_smooth = savgol_filter(f_norm, window_length=window_length, polyorder=3)
    
    # Первая производная
    df_dB = np.gradient(f_smooth, B_loc)
    
    # Вторая производная
    d2f_dB2 = np.gradient(df_dB, B_loc)
    
    derivatives_1st.append((temp, B_loc, df_dB))
    derivatives_2nd.append((temp, B_loc, d2f_dB2))

# =============================================================================
# ГРАФИК 1: ВСЕ ПЕРВЫЕ ПРОИЗВОДНЫЕ на одном графике
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for (temp, B_loc, df_dB), color in zip(derivatives_1st, colors):
    ax.plot(B_loc, df_dB, color=color, lw=1.8, alpha=0.85)

# Оформление
ax.set_xlim(0, 0.25)
ax.set_xlabel(r"$B_{\mathrm{loc}}$, Тл", fontsize=13)
ax.set_ylabel(r"$df/dB_{\mathrm{loc}}$, усл. ед./Тл", fontsize=13)
ax.set_title("Первые производные распределений локальных полей", fontsize=15, pad=15)

# Стиль осей
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# Сетка
ax.grid(True, which='major', lw=0.6, alpha=0.4)
ax.grid(True, which='minor', lw=0.3, alpha=0.2)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Цветовая шкала (легенда по температуре)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
cbar.set_label("Температура, К", fontsize=12)

plt.tight_layout()
plt.show()

# =============================================================================
# ГРАФИК 2: ВСЕ ВТОРЫЕ ПРОИЗВОДНЫЕ на одном графике
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for (temp, B_loc, d2f_dB2), color in zip(derivatives_2nd, colors):
    ax.plot(B_loc, d2f_dB2, color=color, lw=1.8, alpha=0.85)

# Оформление
ax.set_xlim(0, 0.25)
ax.set_xlabel(r"$B_{\mathrm{loc}}$, Тл", fontsize=13)
ax.set_ylabel(r"$d^2f/dB_{\mathrm{loc}}^2$, усл. ед./Тл$^2$", fontsize=13)
ax.set_title("Вторые производные распределений локальных полей", fontsize=15, pad=15)

# Стиль осей
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# Сетка
ax.grid(True, which='major', lw=0.6, alpha=0.4)
ax.grid(True, which='minor', lw=0.3, alpha=0.2)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Цветовая шкала (легенда по температуре)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
cbar.set_label("Температура, К", fontsize=12)

plt.tight_layout()
plt.show()

print("✅ Готово! Построены графики всех первых и вторых производных.")