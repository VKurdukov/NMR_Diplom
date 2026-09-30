import os
import re
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# НАСТРОЙКИ ШРИФТОВ И МАСШТАБИРОВАНИЯ
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

    y_norm = y_vals / np.max(y_vals)

    centroid = np.sum(x_vals * y_norm) / np.sum(y_norm)

    max_idx = np.argmax(y_norm)
    x_max = x_vals[max_idx]

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

# =============================================================================
# НАСТРОЙКИ ОСИ X: ШАГ 2 ГРАДУСА
# =============================================================================
xticks = np.arange(np.floor(temps.min()), np.ceil(temps.max()) + 1, 2)
xtick_labels = ['%d' % int(t) for t in xticks]

# =============================================================================
# График 1: Среднее локальное поле от температуры
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1.plot(temps, centroid_list, 'o-', color='tab:blue', linewidth=2, markersize=6)

ax1.axvline(x=T_N1, color='red', linestyle='--', linewidth=1.5)
ax1.axvline(x=T_N2, color='purple', linestyle='--', linewidth=1.5)

# Подписи TN1 и TN2 рядом с вертикальными линиями (в координатах осей)
ax1.text(T_N1, 0.95, r'$T_{N1}$', color='red', fontsize=12 * FONT_SCALE,
         ha='right', va='top', transform=ax1.get_xaxis_transform())
ax1.text(T_N2, 0.95, r'$T_{N2}$', color='purple', fontsize=12 * FONT_SCALE,
         ha='left', va='top', transform=ax1.get_xaxis_transform())

ax1.set_xlabel('Температура $T$, К')
ax1.set_ylabel(r'Среднее локальное поле $\langle B_{\mathrm{loc}} \rangle$, Тл')

ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)
ax1.grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()
plt.savefig('fig4_2a_spectral_mean.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# График 2: Ширина распределения локальных полей от температуры
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2.plot(temps, std_max_list, 's-', color='tab:red', linewidth=2, markersize=6)

ax2.axvline(x=T_N1, color='red', linestyle='--', linewidth=1.5)
ax2.axvline(x=T_N2, color='purple', linestyle='--', linewidth=1.5)

# Подписи TN1 и TN2 рядом с вертикальными линиями
ax2.text(T_N1, 0.95, r'$T_{N1}$', color='red', fontsize=12 * FONT_SCALE,
         ha='right', va='top', transform=ax2.get_xaxis_transform())
ax2.text(T_N2, 0.95, r'$T_{N2}$', color='purple', fontsize=12 * FONT_SCALE,
         ha='left', va='top', transform=ax2.get_xaxis_transform())

ax2.set_xlabel('Температура $T$, К')
ax2.set_ylabel(r'Ширина распределения $\Delta B$, Тл')

ax2.set_xticks(xticks)
ax2.set_xticklabels(xtick_labels)
ax2.grid(True, linestyle='--', alpha=0.5)

fig2.tight_layout()
plt.savefig('fig4_2b_spectral_std.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Готово! Сохранено: fig4_2a_spectral_mean.png и fig4_2b_spectral_std.png")