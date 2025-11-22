import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator

#Папка с файлами f_Bloc.txt
folder = Path("final_data")

spectra = []

#ЗАГРУЗКА ВСЕХ СПЕКТРОВ
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
    raise RuntimeError("Нет корректных данных f(B_loc) в папке 'final'")

spectra.sort(key=lambda x: x[0])
temps = [t for t, _, _ in spectra]

cmap = LinearSegmentedColormap.from_list("blue_red", ["#C6DCE9", "#FF0000"])
norm = Normalize(vmin=min(temps), vmax=max(temps))
colors = [cmap(norm(t)) for t in temps]


fig, ax = plt.subplots(figsize=(10, 6))

for (temp, B_loc, f_norm), color in zip(spectra, colors):
    ax.plot(B_loc, f_norm, color=color, lw=2.2, label=f"{temp:.2f} K")


ax.set_xlim(0, 0.25)     
ax.set_ylim(bottom=0)     
ax.set_xlabel(r"$B_{\mathrm{loc}}$ (T)", fontsize=13)
ax.set_ylabel(r"$f_{\mathrm{norm}}(B_{\mathrm{loc}})$", fontsize=13)
ax.set_title("Нормированные распределения локальных полей", fontsize=15, pad=15)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

ax.grid(True, which='major', lw=0.6, alpha=0.4)
ax.grid(True, which='minor', lw=0.3, alpha=0.2)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
cbar.set_label("Температура (K)", fontsize=12)

plt.tight_layout()
plt.show()
