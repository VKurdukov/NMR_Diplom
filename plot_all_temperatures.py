import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator

# =============================================================================
# 🔧 КОНФИГУРАЦИЯ
# =============================================================================

# Масштаб шрифта (2.0 = в 2 раза крупнее)
FONT_SCALE = 1.2

# Папка с данными
FOLDER = Path("bootstrap_results")

# Параметр вертикального сдвига кривых (0 = без сдвига)
OFFSET_PER_K = 0

# Диапазон оси X
B_LOC_MIN, B_LOC_MAX = 0, 0.20

# =============================================================================
# 🎨 НАСТРОЙКИ ШРИФТОВ И ОФОРМЛЕНИЯ
# =============================================================================
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams.update({
    'font.size': 10 * FONT_SCALE,
    'axes.titlesize': 14 * FONT_SCALE,
    'axes.labelsize': 12 * FONT_SCALE,
    'xtick.labelsize': 10 * FONT_SCALE,
    'ytick.labelsize': 10 * FONT_SCALE,
    'legend.fontsize': 10 * FONT_SCALE,
})

# =============================================================================
# 🔁 ФУНКЦИЯ НОРМИРОВКИ (только на максимум)
# =============================================================================
def normalize_spectrum(B_loc, f):
    """
    Нормировка спектра на максимальное значение.
    
    Returns
    -------
    f_norm : ndarray
        Нормированный сигнал (max|f_norm| = 1).
    norm_factor : float
        Значение максимума исходного сигнала (для отладки).
    """
    max_val = np.max(np.abs(f))
    if max_val < 1e-10:
        return None, None
    # Сохраняем знак максимума
    sign = np.sign(f[np.argmax(np.abs(f))])
    return f / (sign * max_val), sign * max_val

# =============================================================================
# 📥 ЗАГРУЗКА ДАННЫХ
# =============================================================================
spectra = []

for file in FOLDER.glob("*_f_Bloc.txt"):
    name = file.stem  # например: "3.50K_f_Bloc"
    temp_str = name.replace("_f_Bloc", "")
    match = re.search(r"(\d+\.?\d*)", temp_str)
    if not match:
        print(f"⚠️ Не удалось извлечь температуру из {file.name}")
        continue
    temp = float(match.group(1))

    try:
        data = np.loadtxt(file, skiprows=3)
        B_loc = data[:, 0]
        f = data[:, 1]

        f_norm, norm_factor = normalize_spectrum(B_loc, f)
        
        if f_norm is None:
            print(f"⚠️ Пропущен {file.name}: некорректный коэффициент нормировки")
            continue

        spectra.append((temp, B_loc, f_norm, norm_factor))
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке {file.name}: {e}")
        continue

if not spectra:
    raise RuntimeError(f"Нет корректных данных в папке '{FOLDER}'")

# Сортировка по температуре
spectra.sort(key=lambda x: x[0])
temps = [t for t, _, _, _ in spectra]

# =============================================================================
# 🎨 ПОСТРОЕНИЕ ГРАФИКА
# =============================================================================
cmap = LinearSegmentedColormap.from_list("blue_red", ["#00446C", "#FF0000"])
norm_colors = Normalize(vmin=min(temps), vmax=max(temps))
colors = [cmap(norm_colors(t)) for t in temps]

# Подготовка ylim с учётом сдвига
shifted_maxes = []
for temp, B_loc, f_norm, _ in spectra:
    shifted = f_norm + temp * OFFSET_PER_K
    shifted_maxes.append(np.max(shifted))
y_max = max(shifted_maxes) * 1.05

fig, ax = plt.subplots(figsize=(10, 6))

for (temp, B_loc, f_norm, _), color in zip(spectra, colors):
    f_shifted = f_norm + temp * OFFSET_PER_K
    ax.plot(B_loc, f_shifted, color=color, lw=2.2)

# Оформление осей
ax.set_xlim(B_LOC_MIN, B_LOC_MAX)
ax.set_ylim(bottom=0, top=y_max)

# Подписи осей и заголовок
ax.set_xlabel(r"$B_{\mathrm{loc}}$, Тл")
ax.set_ylabel(r"Относительная интенсивность $f/f_{\mathrm{max}}$")
ax.set_title("Нормированные распределения локальных полей", pad=15)

# Стиль осей
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Сетка
ax.grid(True, which='major', lw=0.6, alpha=0.4)
ax.grid(True, which='minor', lw=0.3, alpha=0.2)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Цветовая шкала (легенда по температуре)
sm = ScalarMappable(cmap=cmap, norm=norm_colors)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
cbar.set_label("Температура, К")

plt.tight_layout()
plt.show()

# =============================================================================
# 📊 ИНФОРМАЦИЯ В КОНСОЛЬ
# =============================================================================
print(f"\n✅ Готово! Обработано {len(spectra)} спектров.")
print(f"📈 Температурный диапазон: {min(temps):.2f} – {max(temps):.2f} К")
print(f"📐 Нормировка: max|f| = 1 (пик = 1)")