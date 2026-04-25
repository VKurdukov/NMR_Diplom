import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Настройки шрифтов и LaTeX для корректного отображения формул
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'DejaVu Sans'

# Температуры Нееля
T_N2 = 9.95  # K, верхний переход
T_N1 = 8.17  # K, нижний переход

folder_path = r"C:\Users\Владимир\Desktop\NMR_Diplom\final_gauss_data"

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

temps = []
centroid_list = []
std_max_list = []

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

    # Нормировка на максимальное значение
    y_norm = y_vals / np.max(y_vals)

    # Центр масс (среднее взвешенное локального поля)
    centroid = np.sum(x_vals * y_norm) / np.sum(y_norm)

    # Индекс максимального значения
    max_idx = np.argmax(y_norm)
    x_max = x_vals[max_idx]

    # Стандартное отклонение от точки с максимальным значением (ширина распределения)
    std_max = np.sqrt(np.sum(y_norm * (x_vals - x_max)**2) / np.sum(y_norm))

    temps.append(temp)
    centroid_list.append(centroid)
    std_max_list.append(std_max)

# Сортировка по температуре
temps = np.array(temps)
centroid_list = np.array(centroid_list)
std_max_list = np.array(std_max_list)

sort_idx = np.argsort(temps)
temps = temps[sort_idx]
centroid_list = centroid_list[sort_idx]
std_max_list = std_max_list[sort_idx]

# Настройки оси X: шаг 1 К
xticks = np.arange(np.floor(temps.min()), np.ceil(temps.max()) + 1, 1)
xtick_labels = ['%d К' % t for t in xticks]

# =============================================================================
# График 1: Среднее локальное поле от температуры
# =============================================================================
plt.figure(figsize=(9, 6))

plt.plot(temps, centroid_list, 'o-', color='tab:blue', linewidth=2, markersize=6, label=r'$\langle B_{\mathrm{loc}} \rangle$')

# Вертикальные линии с корректным форматированием индексов
plt.axvline(x=T_N1, color='red', linestyle='--', linewidth=1.5, label=r'$T_{\mathrm{N1}}$ = %.2f К' % T_N1)
plt.axvline(x=T_N2, color='purple', linestyle='--', linewidth=1.5, label=r'$T_{\mathrm{N2}}$ = %.2f К' % T_N2)

# === ИСПРАВЛЕНО: подписи осей для распределения локальных полей ===
plt.xlabel('Температура $T$, К', fontsize=11)
plt.ylabel(r'Среднее локальное поле $\langle B_{\mathrm{loc}} \rangle$, Тл', fontsize=11)

# === ИСПРАВЛЕНО: только заголовок над графиком, без общего лейбла ===
plt.title('Центр масс распределения полей от температуры', fontsize=12, pad=15)

plt.xticks(xticks, xtick_labels, fontsize=9)
plt.legend(fontsize=10, loc='best', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# =============================================================================
# График 2: Ширина распределения локальных полей от температуры
# =============================================================================
plt.figure(figsize=(9, 6))

# === ИСПРАВЛЕНО: ΔB — ширина распределения, а не дисперсия ===
plt.plot(temps, std_max_list, 's-', color='tab:red', linewidth=2, markersize=6, label=r'$\Delta B$')

plt.axvline(x=T_N1, color='red', linestyle='--', linewidth=1.5, label=r'$T_{\mathrm{N1}}$ = %.2f К' % T_N1)
plt.axvline(x=T_N2, color='purple', linestyle='--', linewidth=1.5, label=r'$T_{\mathrm{N2}}$ = %.2f К' % T_N2)

plt.xlabel('Температура $T$, К', fontsize=11)
# === ИСПРАВЛЕНО: "Ширина", а не "Дисперсия"; ΔB вместо σ_B ===
plt.ylabel(r'Ширина распределения $\Delta B$, Тл', fontsize=11)

plt.title('Дисперсия распределения локальных полей от температуры', fontsize=12, pad=15)

plt.xticks(xticks, xtick_labels, fontsize=9)
plt.legend(fontsize=10, loc='best', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

print("✅ Готово! Построено 2 графика.")