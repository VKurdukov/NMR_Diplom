import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  # ✅ ИСПРАВЛЕНО: был plt, должно быть Path

# Настройки шрифтов и LaTeX для корректного отображения формул
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============ КОНФИГУРАЦИЯ ============
DATA_DIR = "test_data"  # Папка с данными
T_N2 = 9.95  # K, верхний переход
T_N1 = 8.17  # K, нижний переход

# Параметры эксперимента для подписей
EXPERIMENT_LABEL = r'ЯМР $\mathrm{LiCuFe_2(VO_4)_3}$ на ядрах ${}^7\mathrm{Li}$'

# ============ ФУНКЦИИ ============

def extract_temperature(filename: str):
    """Извлекает температуру из названия файла (например, FieldSweep 9.00K.txt → 9.00)"""
    try:
        match = re.search(r'(\d+[\.,]?\d*)K', filename, re.IGNORECASE)
        return float(match.group(1).replace(',', '.')) if match else None
    except Exception as e:
        print(f"Ошибка извлечения температуры: {e}")
        return None

def read_data(filepath):
    """
    Чтение данных из файла FieldSweep.
    Автоматически определяет начало данных (после заголовка).
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Ищем строку с заголовком "Field" — данные начинаются после неё
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('Field') and 'Integral' in line:
                data_start = i + 1
                break
        
        # Парсим данные
        for line in lines[data_start:]:
            line = line.strip().replace(',', '.')
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    field = float(parts[0])      # Field (T или кОе)
                    integral = float(parts[1])   # Integral
                    data.append((field, integral))
                except ValueError:
                    continue
    
    except Exception as e:
        print(f"❌ Ошибка чтения {filepath}: {e}")
    
    return np.array(data) if data else np.array([])

def interactive_noise_selection(x_data, y_data, filename):
    """Интерактивный выбор границ сигнала для обрезки"""
    plt.figure(figsize=(12, 6))
    plt.plot(x_data, y_data, 'b-', linewidth=2, label='Данные')
    
    plt.title(rf"Выбор границ сигнала ({filename}):\n"
              r"ЛКМ — левая граница | ПКМ — правая граница | Enter — подтвердить",
              fontsize=10)
    
    plt.xlabel(r'Поле $B$, Тл')
    plt.ylabel(r'Нормированный сигнал (усл. ед.)')
    plt.grid(True, alpha=0.3)
    
    selected_points = []
    
    def on_click(event):
        if event.inaxes != plt.gca():
            return
        if event.button == 1:  # Левая кнопка - левая граница
            selected_points.append(event.xdata)
            plt.axvline(event.xdata, color='r', linestyle='--', alpha=0.7, linewidth=2)
            print(f"✓ Левая граница: {event.xdata:.4f}")
        elif event.button == 3:  # Правая кнопка - правая граница
            selected_points.append(event.xdata)
            plt.axvline(event.xdata, color='m', linestyle='--', alpha=0.7, linewidth=2)
            print(f"✓ Правая граница: {event.xdata:.4f}")
        plt.draw()
    
    def on_key(event):
        if event.key == 'enter':
            plt.close()
    
    plt.connect('button_press_event', on_click)
    plt.connect('key_press_event', on_key)
    plt.show()
    
    if len(selected_points) >= 2:
        return sorted(selected_points[:2])
    else:
        print("⚠️  Границы не выбраны! Использую автоматические (10% от max).")
        threshold = np.max(y_data) * 0.1
        mask = y_data >= threshold
        if np.any(mask):
            indices = np.where(mask)[0]
            return [x_data[indices[0]], x_data[indices[-1]]]
        else:
            x_min, x_max = np.min(x_data), np.max(x_data)
            return [x_min + 0.2 * (x_max - x_min), x_min + 0.8 * (x_max - x_min)]

def calculate_variance_error(x, y_values, perturbation_fraction=0.05):
    """Оценивает погрешность дисперсии методом конечных разностей."""
    weights = np.abs(y_values)
    if np.sum(weights) == 0:
        return 0
    
    mean_nom = np.average(x, weights=weights)
    delta = perturbation_fraction * np.max(np.abs(y_values))
    
    # Сдвиг вниз
    y_low = np.clip(y_values - delta, 0, None)
    weights_low = np.abs(y_low)
    if np.sum(weights_low) > 0:
        mean_low = np.average(x, weights=weights_low)
        var_low = np.average((x - mean_low) ** 2, weights=weights_low)
    else:
        var_low = 0
    
    # Сдвиг вверх
    y_high = y_values + delta
    weights_high = np.abs(y_high)
    mean_high = np.average(x, weights=weights_high)
    var_high = np.average((x - mean_high) ** 2, weights=weights_high)
    
    err_var = np.abs(var_high - var_low) / 2.0
    return err_var

def calculate_std_error(var, err_var):
    """Оценивает ошибку корня из дисперсии через метод переноса ошибок."""
    if var <= 0:
        return 0
    std = np.sqrt(var)
    if std == 0:
        return 0
    return abs(1 / (2 * std)) * err_var

def calculate_stats(x_data, y_data, noise_var):
    """Расчет статистик с погрешностями"""
    weights = np.abs(y_data)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return None
    
    max_value = np.max(y_data)
    max_index = np.argmax(y_data)
    max_x = x_data[max_index]
    mean_val = np.average(x_data, weights=weights)
    variance = np.average((x_data - mean_val) ** 2, weights=weights)
    
    n = len(x_data)
    dx = np.mean(np.diff(x_data)) if len(x_data) > 1 else 0.01
    
    err_max_x = abs(dx) / 2
    err_mean = abs(dx) / (2 * np.sqrt(n)) if n > 0 else 0
    err_max_value = np.sqrt(noise_var) if noise_var > 0 else 0
    err_var = calculate_variance_error(x_data, y_data, perturbation_fraction=0.05)
    
    std_dev = np.sqrt(variance) if variance > 0 else 0
    err_std = calculate_std_error(variance, err_var)
    
    stats = {
        'max_value': max_value,
        'max_field': max_x,
        'mean_field': mean_val,
        'variance': variance,
        'std_dev': std_dev,
        'noise_var': noise_var,
        'err_max_field': err_max_x,
        'err_mean': err_mean,
        'err_var': err_var,
        'err_max_value': err_max_value,
        'err_std': err_std
    }
    
    return stats

def process_file(filepath, temp):
    """Обработка файла с интерактивным выбором границ"""
    try:
        data = read_data(filepath)
        if data.size == 0:
            print(f"❌ Пустой файл или ошибка парсинга: {filepath}")
            return None
        
        x_data = data[:, 0]
        y_data = data[:, 1]
        
        print(f"\n📊 Обработка: {Path(filepath).name} (T = {temp:.2f} K)")
        print(f"   Точек данных: {len(x_data)}")
        print(f"   Диапазон поля: {x_data.min():.4f} – {x_data.max():.4f} Тл")
        print(f"   Max сигнала: {y_data.max():.2f}")
        
        bounds = interactive_noise_selection(x_data, y_data, Path(filepath).name)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_data, y_data, 'b-', linewidth=2, label='Исходные данные')
        plt.axvspan(x_data.min(), bounds[0], color='r', alpha=0.2, label='Левый шум')
        plt.axvspan(bounds[1], x_data.max(), color='m', alpha=0.2, label='Правый шум')
        plt.axvline(bounds[0], color='r', linestyle='--', linewidth=2)
        plt.axvline(bounds[1], color='m', linestyle='--', linewidth=2)
        
        plt.title(rf"{Path(filepath).name} — Обрезанная область "
                  rf"({bounds[0]:.4f} – {bounds[1]:.4f} Тл)",
                  fontsize=11)
        
        plt.xlabel(r'Поле $B$, Тл')
        plt.ylabel(r'Нормированный сигнал (усл. ед.)')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        peak_mask = (x_data >= bounds[0]) & (x_data <= bounds[1])
        noise_mask = ~peak_mask
        
        peak_x = x_data[peak_mask]
        peak_y = y_data[peak_mask]
        noise_y = y_data[noise_mask]
        
        if len(peak_x) == 0:
            print(f"⚠️  Нет данных в выбранной области!")
            return None
        
        noise_var = np.mean(noise_y ** 2) if len(noise_y) > 0 else 0
        
        stats = calculate_stats(peak_x, peak_y, noise_var)
        if stats:
            stats['temperature'] = temp
        
        return stats
    
    except Exception as e:
        print(f"❌ Ошибка обработки {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============ ОСНОВНОЙ КОД ============

if __name__ == "__main__":
    txt_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
    
    if not txt_files:
        print(f"❌ Нет txt файлов в папке {DATA_DIR}")
        exit(1)
    
    print(f"✅ Найдено {len(txt_files)} файлов в {DATA_DIR}")
    
    all_stats = []
    
    for filename in txt_files:
        filepath = os.path.join(DATA_DIR, filename)
        temp = extract_temperature(filename)
        
        if temp is None:
            print(f"⚠️  Пропущен {filename}: не найдена температура")
            continue
        
        stats = process_file(filepath, temp)
        
        if stats:
            all_stats.append(stats)
    
    if all_stats:
        print(f"\n{'='*50}")
        print(f"✅ Обработано {len(all_stats)} файлов")
        
        all_stats.sort(key=lambda s: s['temperature'])
        
        temps = np.array([s['temperature'] for s in all_stats])
        mean_fields = np.array([s['mean_field'] for s in all_stats])
        err_means = np.array([s['err_mean'] for s in all_stats])
        std_devs = np.array([s['std_dev'] for s in all_stats])
        err_stds = np.array([s['err_std'] for s in all_stats])
        
        # === ГРАФИК 1: Средняя позиция пика ===
        plt.figure(figsize=(10, 6))
        plt.errorbar(temps, mean_fields, yerr=err_means, 
                     marker='o', linestyle='-', color='tab:blue',
                     linewidth=2, markersize=6, capsize=4, capthick=1.5, 
                     label=r'$\langle B \rangle$')
        
        plt.axvline(T_N1, color='orange', linestyle='--', linewidth=1.5, 
                    label=r'$T_{\mathrm{N1}} = %.2f$~К' % T_N1)
        plt.axvline(T_N2, color='olive', linestyle='--', linewidth=1.5, 
                    label=r'$T_{\mathrm{N2}} = %.2f$~К' % T_N2)
        
        plt.xlabel(r'Температура $T$, К', fontsize=11)
        plt.ylabel(r'Среднее поле $\langle B \rangle$, Тл', fontsize=11)
        
        plt.title(EXPERIMENT_LABEL + '\n' + r'Среднее положение пика от температуры', 
                  fontsize=11, pad=15)
        
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
        
        # === ГРАФИК 2: Стандартное отклонение ===
        plt.figure(figsize=(10, 6))
        plt.errorbar(temps, std_devs, yerr=err_stds, 
                     marker='s', linestyle='-', color='tab:red',
                     linewidth=2, markersize=6, capsize=4, capthick=1.5, 
                     label=r'$\sigma_B$')
        
        plt.axvline(T_N1, color='orange', linestyle='--', linewidth=1.5, 
                    label=r'$T_{\mathrm{N1}} = %.2f$~К' % T_N1)
        plt.axvline(T_N2, color='olive', linestyle='--', linewidth=1.5, 
                    label=r'$T_{\mathrm{N2}} = %.2f$~К' % T_N2)
        
        plt.xlabel(r'Температура $T$, К', fontsize=11)
        plt.ylabel(r'Стандартное отклонение $\sigma_B$, Тл', fontsize=11)
        
        plt.title(EXPERIMENT_LABEL + '\n' + r'Ширина распределения локальных полей', 
                  fontsize=11, pad=15)
        
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
        
        # === ИСПРАВЛЕНО: используем rf-строки для формул с переменными ===
        print("\n📊 Статистика результатов:")
        print(f"   Температурный диапазон: {temps.min():.2f} – {temps.max():.2f} К")
        print(rf"   $\langle B \rangle$: {mean_fields.mean():.6f} ± {mean_fields.std():.6f} Тл")
        print(rf"   $\sigma_B$: {std_devs.mean():.6f} ± {std_devs.std():.6f} Тл")
    else:
        print("❌ Не удалось обработать ни один файл")

print("\n✅ Готово!")