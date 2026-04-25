import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter

# Настройки шрифтов и LaTeX
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'DejaVu Sans'

# Температуры Нееля
T_N2 = 9.95  # K, верхний переход
T_N1 = 8.17  # K, нижний переход

# Папка с файлами
folder = Path("final_gauss_data")

spectra = []

# === ЗАГРУЗКА ВСЕХ СПЕКТРОВ ===
for file in folder.glob("*_f_Bloc.txt"):
    name = file.stem
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

        integral = np.trapezoid(f, B_loc)
        if integral <= 0:
            print(f"⚠️ Пропущен {file.name}: интеграл = {integral:.2e} ≤ 0")
            continue

        f_norm = f / integral
        spectra.append((temp, B_loc, f_norm))
    except Exception as e:
        print(f"❌ Ошибка при загрузке {file.name}: {e}")
        continue

if not spectra:
    raise RuntimeError("Нет корректных данных f(B_loc) в папке 'final_gauss_data'")

# Сортируем по температуре
spectra.sort(key=lambda x: x[0])

# === ВЫЧИСЛЕНИЕ ПЕРВЫХ ПРОИЗВОДНЫХ ===
derivatives_1st = []

for temp, B_loc, f_norm in spectra:
    window_length = min(11, len(f_norm) // 2 * 2 + 1)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1
    
    f_smooth = savgol_filter(f_norm, window_length=window_length, polyorder=3)
    df_dB = np.gradient(f_smooth, B_loc)
    
    derivatives_1st.append((temp, B_loc, df_dB))

# =============================================================================
# ИНТЕРАКТИВНЫЙ ВЫБОР ТОЧЕК ДЛЯ КАЖДОЙ ТЕМПЕРАТУРЫ
# =============================================================================

all_selected_points = {}  # {temp: [point1, point2]}

print("\n" + "="*70)
print("📌 ИНСТРУКЦИЯ:")
print("   Для каждой температуры:")
print("   1. Кликните ЛКМ на первую точку (положительный пик)")
print("   2. Кликните ЛКМ на вторую точку (отрицательный пик)")
print("   3. Нажмите Enter для перехода к следующей температуре")
print("="*70 + "\n")

for idx, (temp, B_loc, df_dB) in enumerate(derivatives_1st):
    print(f"\n📊 Обработка: T = {temp:.2f} К ({idx+1}/{len(derivatives_1st)})")
    
    selected_points = []
    point_markers = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(B_loc, df_dB, 'b-', linewidth=2, label=f'T = {temp:.2f} К')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(0, 0.25)
    ax.set_xlabel(r'$B_{\mathrm{loc}}$, Тл', fontsize=11)
    ax.set_ylabel(r'$df/dB_{\mathrm{loc}}$, усл. ед./Тл', fontsize=11)
    ax.set_title(f'Первая производная при T = {temp:.2f} К\n'
                 f'Выберите 2 точки (ЛКМ), затем нажмите Enter', fontsize=12, pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # ЛКМ
            selected_points.append(event.xdata)
            marker = ax.axvline(event.xdata, color='red', linestyle='--', 
                               linewidth=2, alpha=0.8)
            point_markers.append(marker)
            ax.text(event.xdata, ax.get_ylim()[1]*0.95, 
                   f'{event.xdata:.4f}', ha='center', va='top',
                   fontsize=10, color='red', fontweight='bold')
            print(f"   ✓ Точка {len(selected_points)}: B_loc = {event.xdata:.6f} Тл")
            fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key == 'enter':
            if len(selected_points) >= 2:
                print(f"   ✅ Выбрано 2 точки. Переход к следующей температуре...")
                plt.close(fig)
            else:
                print(f"   ⚠️ Выберите ещё {2 - len(selected_points)} точку(и)!")
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()
    
    if len(selected_points) >= 2:
        all_selected_points[temp] = selected_points[:2]
    else:
        print(f"   ⚠️ Пропущено T = {temp:.2f} К: выбрано только {len(selected_points)} точек")

# =============================================================================
# ПОСТРОЕНИЕ ТЕМПЕРАТУРНОЙ ЗАВИСИМОСТИ
# =============================================================================

if all_selected_points:
    temps_plot = sorted(all_selected_points.keys())
    point1_values = [all_selected_points[t][0] for t in temps_plot]
    point2_values = [all_selected_points[t][1] for t in temps_plot]
    
    print(f"\n{'='*70}")
    print("📊 ТЕМПЕРАТУРНАЯ ЗАВИСИМОСТЬ ВЫБРАННЫХ ТОЧЕК")
    print(f"{'='*70}")
    print(f"Обработано температур: {len(temps_plot)}")
    print(f"{'='*70}\n")
    
    # === ГРАФИК ТЕМПЕРАТУРНОЙ ЗАВИСИМОСТИ ===
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    ax2.plot(temps_plot, point1_values, 'o-', color='tab:blue', 
             linewidth=2, markersize=6, label=r'Точка 1 (положит. пик)')
    ax2.plot(temps_plot, point2_values, 's-', color='tab:red', 
             linewidth=2, markersize=6, label=r'Точка 2 (отриц. пик)')
    ax2.plot(temps_plot, np.abs(np.array(point2_values) - np.array(point1_values)), 
             '^-', color='tab:green', linewidth=2, markersize=6, label=r'$\Delta B$')
    
    # Вертикальные линии температур Нееля
    ax2.axvline(x=T_N1, color='orange', linestyle='--', linewidth=1.5, 
                label=r'$T_{\mathrm{N1}}$ = %.2f К' % T_N1, alpha=0.8)
    ax2.axvline(x=T_N2, color='purple', linestyle='--', linewidth=1.5, 
                label=r'$T_{\mathrm{N2}}$ = %.2f К' % T_N2, alpha=0.8)
    
    ax2.set_xlabel('Температура $T$, К', fontsize=12)
    ax2.set_ylabel(r'$B_{\mathrm{loc}}$, Тл', fontsize=12)
    ax2.set_title('Температурная зависимость положений пиков производной', 
                  fontsize=13, pad=15)
    
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    
    # Настройки оси X
    xticks = np.arange(np.floor(min(temps_plot)), 
                       np.ceil(max(temps_plot)) + 1, 1)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(['%d К' % t for t in xticks], fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Сохранение данных
    output_file = "selected_points_temperature_dependence.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Temperature (K)\tPoint1 (T)\tPoint2 (T)\tDelta B (T)\n")
        for t, p1, p2 in zip(temps_plot, point1_values, point2_values):
            f.write(f"{t:.2f}\t{p1:.6f}\t{p2:.6f}\t{abs(p2-p1):.6f}\n")
    
    print(f"✅ Данные сохранены в файл: {output_file}")
    print("✅ Готово! Построен график температурной зависимости.")
else:
    print("⚠️ Не выбрано ни одной пары точек!")