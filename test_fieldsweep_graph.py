import os
import re
import numpy as np
import matplotlib.pyplot as plt


folder_path = r"C:\Users\Владимир\Desktop\Diplom\test_data"


txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]


data_list = []
for filename in txt_files:
    filepath = os.path.join(folder_path, filename)
    
    match = re.search(r'(\d+\.\d+)K', filename)
    if not match:
        print(f"⚠️ Не удалось извлечь T из {filename}")
        continue
    temp = float(match.group(1))
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    x_vals, y_vals = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except ValueError:
                continue

    if not x_vals:
        continue

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    data_list.append((temp, x_vals, y_vals))

data_list.sort(key=lambda t: t[0])

temps = [t[0] for t in data_list]
mean_x_list = []
var_x_list = []

plt.figure(figsize=(10, 6))

cmap = plt.cm.turbo  
colors = cmap(np.linspace(0, 1, len(data_list)))

for (temp, x_vals, y_vals), color in zip(data_list, colors):
    y_max = np.max(np.abs(y_vals))
    if y_max != 0:
        y_vals = y_vals / y_max

    plt.plot(x_vals, y_vals, label=f'{temp} K', color=color)
    
    sum_y = np.sum(y_vals)
    if np.isclose(sum_y, 0):
        continue
    mean_x = np.sum(x_vals * y_vals) / sum_y
    var_x = np.sum(y_vals * (x_vals - mean_x)**2) / sum_y

    mean_x_list.append(mean_x)
    var_x_list.append(var_x)

plt.xlabel('Поле (X)')
plt.ylabel('Нормированный сигнал (Y)')
plt.title('FieldSweep зависимости (нормированные функции)')
plt.legend(title="Температура", loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

#График <x> от T
temps = np.array(temps)
mean_x_list = np.array(mean_x_list)
var_x_list = np.array(var_x_list)

plt.figure(figsize=(8, 5))
plt.plot(temps, mean_x_list, 'o-', color='tab:blue')
plt.xlabel('Температура (K)')
plt.ylabel(r'$\langle x \rangle$')
plt.title('Среднее по X vs Температура')
plt.grid(True)
plt.tight_layout()
plt.show()

#График дисперсии от T
plt.figure(figsize=(8, 5))
plt.plot(temps, var_x_list, 's-', color='tab:red')
plt.xlabel('Температура (K)')
plt.ylabel(r'$\sigma_x^2$')
plt.title('Дисперсия по X vs Температура')
plt.grid(True)
plt.tight_layout()
plt.show()

print("✅ Готово!")
