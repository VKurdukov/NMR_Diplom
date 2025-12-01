import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============ КОНФИГУРАЦИЯ ============
DATA_DIR = "test_data"  # Папка с данными


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
    Ожидает колонки: Field, Integral, Fourier, MaxValue, RST, ...
    Возвращает: (Field, Integral) как основные данные
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
    plt.title(f"Выбор границ сигнала ({filename}):\n"
              f"ЛКМ - левая граница | ПКМ - правая граница | Enter - подтвердить")
    plt.xlabel('Field (T)')
    plt.ylabel('Integral')
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
        # Автоматическое определение пика: область где y > 10% от максимума
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
    """
    Оценивает погрешность дисперсии методом конечных разностей.
    """
    weights = np.abs(y_values)
    if np.sum(weights) == 0:
        return 0
    
    mean_nom = np.average(x, weights=weights)
    
    # Определяем сдвиг
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
    
    # Ошибка дисперсии
    err_var = np.abs(var_high - var_low) / 2.0
    return err_var



def calculate_stats(x_data, y_data, noise_var):
    """Расчет статистик с погрешностями"""
    weights = np.abs(y_data)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return None
    
    # Основные статистики
    max_value = np.max(y_data)
    max_index = np.argmax(y_data)
    max_x = x_data[max_index]
    mean_val = np.average(x_data, weights=weights)
    variance = np.average((x_data - mean_val) ** 2, weights=weights)
    
    # Погрешности
    n = len(x_data)
    dx = np.mean(np.diff(x_data)) if len(x_data) > 1 else 0.01
    
    err_max_x = abs(dx) / 2
    err_mean = abs(dx) / (2 * np.sqrt(n)) if n > 0 else 0
    err_max_value = np.sqrt(noise_var) if noise_var > 0 else 0
    err_var = calculate_variance_error(x_data, y_data, perturbation_fraction=0.05)
    
    stats = {
        'max_value': max_value,
        'max_field': max_x,
        'mean_field': mean_val,
        'variance': variance,
        'noise_var': noise_var,
        'err_max_field': err_max_x,
        'err_mean': err_mean,
        'err_var': err_var,
        'err_max_value': err_max_value
    }
    
    return stats



def process_file(filepath, temp):
    """Обработка файла с интерактивным выбором границ"""
    try:
        data = read_data(filepath)
        if data.size == 0:
            print(f"❌ Пустой файл или ошибка парсинга: {filepath}")
            return None
        
        x_data = data[:, 0]  # Field
        y_data = data[:, 1]  # Integral
        
        print(f"\n📊 Обработка: {Path(filepath).name} (T = {temp:.2f} K)")
        print(f"   Точек данных: {len(x_data)}")
        print(f"   Диапазон Field: {x_data.min():.4f} - {x_data.max():.4f}")
        print(f"   Max Integral: {y_data.max():.2f}")
        
        # Интерактивный выбор границ
        bounds = interactive_noise_selection(x_data, y_data, Path(filepath).name)
        
        # Визуализация обрезанной области
        plt.figure(figsize=(12, 6))
        plt.plot(x_data, y_data, 'b-', linewidth=2, label='Исходные данные')
        plt.axvspan(x_data.min(), bounds[0], color='r', alpha=0.2, label='Левый шум')
        plt.axvspan(bounds[1], x_data.max(), color='m', alpha=0.2, label='Правый шум')
        plt.axvline(bounds[0], color='r', linestyle='--', linewidth=2)
        plt.axvline(bounds[1], color='m', linestyle='--', linewidth=2)
        plt.title(f"{Path(filepath).name} - Обрезанная область ({bounds[0]:.4f} - {bounds[1]:.4f})")
        plt.xlabel('Field (T)')
        plt.ylabel('Integral')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Разделение данных
        peak_mask = (x_data >= bounds[0]) & (x_data <= bounds[1])
        noise_mask = ~peak_mask
        
        peak_x = x_data[peak_mask]
        peak_y = y_data[peak_mask]
        noise_y = y_data[noise_mask]
        
        if len(peak_x) == 0:
            print(f"⚠️  Нет данных в выбранной области!")
            return None
        
        # Расчет дисперсии шума
        noise_var = np.mean(noise_y ** 2) if len(noise_y) > 0 else 0
        
        # Расчет статистик для пика
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
    # Получаем список файлов
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
    
    # Построение финальных графиков
    if all_stats:
        print(f"\n{'='*50}")
        print(f"✅ Обработано {len(all_stats)} файлов")
        
        # Сортируем по температуре
        all_stats.sort(key=lambda s: s['temperature'])
        
        # Извлекаем данные для графиков
        temps = np.array([s['temperature'] for s in all_stats])
        mean_fields = np.array([s['mean_field'] for s in all_stats])
        err_means = np.array([s['err_mean'] for s in all_stats])
        variances = np.array([s['variance'] for s in all_stats])
        err_vars = np.array([s['err_var'] for s in all_stats])
        
        # График 1: Средняя позиция (Mean Field) vs Температура
        plt.figure(figsize=(12, 6))
        plt.errorbar(temps, mean_fields, yerr=err_means, fmt='o-', color='tab:blue', 
                     linewidth=2, markersize=8, capsize=5, capthick=2, label='Mean Field')
        plt.xlabel('Температура (K)', fontsize=12)
        plt.ylabel('Mean Field (T)', fontsize=12)
        plt.title('Средняя позиция пика vs Температура', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
        
        # График 2: Дисперсия (Variance) vs Температура
        plt.figure(figsize=(12, 6))
        plt.errorbar(temps, variances, yerr=err_vars, fmt='s-', color='tab:red', 
                     linewidth=2, markersize=8, capsize=5, capthick=2, label='Variance')
        plt.xlabel('Температура (K)', fontsize=12)
        plt.ylabel('Variance (T²)', fontsize=12)
        plt.title('Дисперсия пика vs Температура', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
        
        # Вывод финальной информации
        print(f"\n📊 Статистика результатов:")
        print(f"   Температурный диапазон: {temps.min():.2f} - {temps.max():.2f} K")
        print(f"   Mean Field: {mean_fields.mean():.6f} ± {mean_fields.std():.6f} T")
        print(f"   Variance: {variances.mean():.6e} ± {variances.std():.6e} T²")
    else:
        print("❌ Не удалось обработать ни один файл")

print("\n✅ Готово!")