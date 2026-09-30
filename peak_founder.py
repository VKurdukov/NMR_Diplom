import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =============================================================================
# 🔤 НАСТРОЙКИ ШРИФТОВ И МАСШТАБИРОВАНИЯ
# =============================================================================
FONT_SCALE = 1.5  # Коэффициент увеличения текста (2.0 = в 2 раза)

plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'DejaVu Sans'

# Применяем масштаб ко всем базовым элементам текста
plt.rcParams.update({
    'font.size': 10 * FONT_SCALE,          # 20
    'axes.titlesize': 12 * FONT_SCALE,     # 24
    'axes.labelsize': 11 * FONT_SCALE,     # 22
    'xtick.labelsize': 10 * FONT_SCALE,    # 20
    'ytick.labelsize': 10 * FONT_SCALE,    # 20
    'legend.fontsize': 10 * FONT_SCALE,    # 20
})

# Температуры Нееля
T_N2 = 9.95  # K, верхний переход
T_N1 = 8.17  # K, нижний переход

folder_path = r"C:\Users\Владимир\Desktop\NMR_Diplom\final_gauss_data"

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

temps = []
first_peak_x = []
second_peak_x = []

for filename in txt_files:
    filepath = os.path.join(folder_path, filename)
    
    match = re.search(r'(\d+\.\d+)K', filename)
    if not match:
        print(f"⚠️ Пропуск: не найдена температура в {filename}")
        continue
    temp = float(match.group(1))
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
    except Exception as e:
        print(f"❌ Ошибка чтения {filename}: {e}")
        continue

    x_vals, y_vals = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except ValueError:
                continue

    if len(x_vals) == 0:
        continue

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    y_for_peaks = y_vals  # предполагаем, что пики — положительные
    y_max = np.max(y_for_peaks)
    if y_max <= 0:
        continue

    peaks, _ = find_peaks(y_for_peaks, prominence=0.01 * y_max, distance=3)
    if len(peaks) == 0:
        continue

    peak_heights = y_for_peaks[peaks]
    sorted_idx = np.argsort(-peak_heights)
    sorted_peaks = peaks[sorted_idx]

    x1 = x_vals[sorted_peaks[0]]
    x2 = x_vals[sorted_peaks[1]] if len(sorted_peaks) >= 2 else x1

    temps.append(temp)
    first_peak_x.append(x1)
    second_peak_x.append(x2)

# Сортировка по температуре
temps = np.array(temps)
first_peak_x = np.array(first_peak_x)
second_peak_x = np.array(second_peak_x)

sort_idx = np.argsort(temps)
temps = temps[sort_idx]
first_peak_x = first_peak_x[sort_idx]
second_peak_x = second_peak_x[sort_idx]

# =============================================================================
# График: Зависимость положения пиков от температуры
# =============================================================================
plt.figure(figsize=(9, 6))  # Размер фигуры не изменён

plt.plot(temps, first_peak_x, '-', label='Первый пик', 
         color='tab:blue', linewidth=2)
plt.plot(temps, second_peak_x, '-', label='Второй пик', 
         color='tab:orange', linewidth=2)

plt.axvline(x=T_N1, color='red', linestyle='--', linewidth=2, 
            label=r'$T_{\mathrm{N1}} = %.2f$ К' % T_N1)
plt.axvline(x=T_N2, color='purple', linestyle='--', linewidth=2, 
            label=r'$T_{\mathrm{N2}} = %.2f$ К' % T_N2)

# ✅ Убраны "К" из меток, fontsize управляется глобально через rcParams
xticks = np.arange(np.floor(temps.min()), np.ceil(temps.max()) + 1, 1)
plt.xticks(xticks, ['%d' % int(t) for t in xticks])

# ✅ Убраны явные fontsize → теперь все подписи автоматически ×2
plt.xlabel('Температура $T$, К')
plt.ylabel('Положение пика, Тл')
plt.title('Зависимость положения пиков от температуры', pad=15)

plt.legend(loc='best', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()