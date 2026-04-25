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

# Параметры эксперимента для подписи
experiment_label = r'ЯМР $\mathrm{LiCuFe_2(VO_4)_3}$ на ядрах ${}^7\mathrm{Li}$' + '\n' + \
                   r'$\nu = 12$ МГц, $B_{\mathrm{L}} = 0.725$ Тл'

# Укажите путь к папке с данными
folder_path = r"C:\Users\Владимир\Desktop\NMR_Diplom\test_data"

# Получаем список .txt файлов
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

data_list = []

for filename in txt_files:
    filepath = os.path.join(folder_path, filename)
    
    # Извлекаем температуру из имени файла (например, "12.5K.txt" → 12.5)
    match = re.search(r'(\d+\.\d+)K', filename)
    if not match:
        print(f"⚠️ Не удалось извлечь температуру из имени файла: {filename}")
        continue
    temp = float(match.group(1))
    
    # Читаем данные (пропускаем первую строку — заголовок)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
    except Exception as e:
        print(f"❌ Ошибка чтения файла {filename}: {e}")
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

    if len(x_vals) == 0 or len(y_vals) == 0:
        print(f"⚠️ Пропуск {filename}: нет данных")
        continue

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Убедимся, что длины совпадают
    if x_vals.shape != y_vals.shape:
        print(f"⚠️ Несовпадение размеров в {filename}, пропуск")
        continue

    data_list.append((temp, x_vals, y_vals))

# Сортируем по температуре
data_list.sort(key=lambda t: t[0])

# Подготовка списков для статистик
temps = []
mean_x_list = []
var_x_list = []

# =============================================================================
# График 1: ЯМР спектры при разных температурах
# =============================================================================
plt.figure(figsize=(20, 9))
cmap = plt.cm.turbo
colors = cmap(np.linspace(0, 1, len(data_list)))
offset_step = 0.0

for i, (temp, x_vals, y_vals) in enumerate(data_list):
    # Нормировка по максимуму модуля
    y_max = np.max(np.abs(y_vals))
    if y_max == 0:
        print(f"⚠️ Пропуск {temp} K: нулевой сигнал")
        continue
    y_norm = y_vals / y_max

    # Веса для вычисления среднего и дисперсии
    weights = np.abs(y_norm)
    total_weight = np.sum(weights)
    if total_weight == 0:
        print(f"⚠️ Пропуск {temp} K: сумма весов = 0")
        continue

    w = weights / total_weight

    mean_x = np.sum(w * x_vals)
    var_x = np.sum(w * (x_vals - mean_x)**2)

    temps.append(temp)
    mean_x_list.append(mean_x)
    var_x_list.append(var_x)

    # Сдвигаем кривую по вертикали для лучшей визуализации
    y_shifted = y_norm + i * offset_step
    plt.plot(x_vals, y_shifted, label=f'{temp:.1f} К', color=colors[i])

# Вертикальная линия
B_L = 0.725
plt.axvline(x=B_L, color='black', linestyle='--', linewidth=1.5, label=r'$B_{\mathrm{L}} = 0.725$ Тл')

# Обрезание по оси X: от 0.5 до 0.95 Тл
plt.xlim(0.5, 0.95)

plt.xlabel(r'Поле $B$, Тл')
plt.ylabel(r'Нормированный сигнал (усл. ед.)')
plt.title(experiment_label + '\n' + 
          rf'Температуры: $T_{{\mathrm{{N1}}}} = {T_N1}$ К, $T_{{\mathrm{{N2}}}} = {T_N2}$ К',
          fontsize=11)
plt.legend(title="Температура", loc='upper right', fontsize='small', ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Преобразуем в массивы
temps = np.array(temps)
mean_x_list = np.array(mean_x_list)
var_x_list = np.array(var_x_list)

# =============================================================================
# График 2: Среднее значение поля от температуры
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(temps, mean_x_list, 'o-', color='tab:blue', markersize=6)
plt.xlabel(r'Температура $T$, К')
plt.ylabel(r'$\langle B \rangle$, Тл')
plt.title(experiment_label + '\n' + 
          rf'Среднее поле, $T_{{\mathrm{{N1}}}} = {T_N1}$ К, $T_{{\mathrm{{N2}}}} = {T_N2}$ К',
          fontsize=10)
# Внимание: здесь НЕ ставим xlim(0.5, 0.95), так как ось X — это температура (~8-12 К)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =============================================================================
# График 3: Дисперсия поля от температуры
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(temps, var_x_list, 's-', color='tab:red', markersize=6)
plt.xlabel(r'Температура $T$, К')
plt.ylabel(r'$\sigma_B^2$, Тл$^2$')
plt.title(experiment_label + '\n' + 
          rf'Дисперсия поля, $T_{{\mathrm{{N1}}}} = {T_N1}$ К, $T_{{\mathrm{{N2}}}} = {T_N2}$ К',
          fontsize=10)
# Внимание: здесь НЕ ставим xlim(0.5, 0.95), так как ось X — это температура (~8-12 К)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("✅ Готово!")