# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла
with open("Статистика.txt", "r", encoding="UTF-8") as file:
    data = list(map(int, file.read().split()))

# Сортируем данные и формируем уникальные возраста и их частоты
ages = list(range(min(data), max(data) + 1))
frequencies = [data.count(age) for age in ages]

# Формируем DataFrame для удобства расчетов
data_df = pd.DataFrame({"Age": ages, "Frequency": frequencies})

# Задание 1: Структурная идентификация
# Функция для расчета средних значений для разных зависимостей
def calculate_structural_means(df):
    x1 = df.iloc[0]["Age"]  # Минимальный возраст
    xn = df.iloc[-1]["Age"]  # Максимальный возраст
    y1 = df.iloc[0]["Frequency"]  # Частота для минимального возраста
    yn = df.iloc[-1]["Frequency"]  # Частота для максимального возраста

    means = {
        "linear": ((x1 + xn) / 2, (y1 + yn) / 2),
        "geometric": (np.sqrt(x1 * xn), np.sqrt(y1 * yn)),
        "harmonic": ((2 * x1 * xn) / (x1 + xn), (2 * y1 * yn) / (y1 + yn)),
        "dependency_4": ((2 * x1 * xn) / (x1 + xn), (y1 + yn) / 2),
        "dependency_5": ((x1 + xn) / 2, (2 * y1 * yn) / (y1 + yn)),
        "dependency_6": ((2 * x1 * xn) / (x1 + xn), (2 * y1 * yn) / (y1 + yn)),
        "dependency_7": (np.sqrt(x1 * xn), (y1 + yn) / 2)
    }
    return means

# Расчет средних значений для каждой зависимости
structural_means = calculate_structural_means(data_df)

# Экспериментальное значение ys (среднее значение частот)
xs = (data_df.iloc[0]["Age"] + data_df.iloc[-1]["Age"]) / 2
y_s = np.interp(xs, data_df["Age"], data_df["Frequency"])

# Расчет отклонений и определение минимального отклонения
absolute_deviations = {key: abs(mean[1] - y_s) for key, mean in structural_means.items()}
min_deviation_key = min(absolute_deviations, key=absolute_deviations.get)

# Расчет процента отклонений
percent_deviations = {key: (dev / sum(frequencies)) * 100 for key, dev in absolute_deviations.items()}

# Вывод результатов для Задания 1
print("\nЗадание 1: Структурная идентификация")
for key, mean in structural_means.items():
    print(f"Зависимость: {key.capitalize()}, Средние значения: ̅x_s = {round(mean[0], 1)}, ̅y_s = {round(mean[1], 1)}, "
          f"Экспериментальное значение y_s = {round(y_s, 1)}, Отклонение Δ_s = {round(absolute_deviations[key], 1)} "
          f"({round(percent_deviations[key], 1)}%)")
print(f"Минимальное отклонение у зависимости: {min_deviation_key.capitalize()}")

# Задание 2: Аппроксимирующий многочлен
# Определение степени многочлена, соответствующей условию
max_degree = 10  # Максимальная степень многочлена для проверки

# Создаем массивы для аппроксимации
x_vals = np.array(data_df["Age"], dtype=np.float64)
y_vals = np.array(data_df["Frequency"], dtype=np.float64)

# Перебираем степени и проверяем разности
best_degree = None
for degree in range(1, max_degree + 1):
    # Построение аппроксимации
    coefficients = np.polyfit(x_vals, y_vals, degree)
    polynomial = np.poly1d(coefficients)

    # Вычисление разностей
    approximations = polynomial(x_vals)
    differences = np.abs(y_vals - approximations)

    # Проверка условия: максимум разности не превышает 2% от суммы частот
    if np.max(differences) <= 0.02 * sum(frequencies):
        best_degree = degree
        break

# Вывод результатов для Задания 2
print("\nЗадание 2: Аппроксимирующий многочлен")
if best_degree:
    print(f"Оптимальная степень аппроксимирующего многочлена: {best_degree}")
else:
    print("Не найден многочлен, удовлетворяющий условию (максимальная разность <= 2% от суммы частот)")

# Построение графика полигона частот
plt.figure(figsize=(10, 6))
plt.plot(data_df["Age"], data_df["Frequency"], label="Полигон частот", marker="o")
plt.title("Полигон частот (Возраст vs Частота)")
plt.xlabel("Возраст")
plt.ylabel("Частота")
plt.legend()
plt.grid()
plt.show()

# Построение аппроксимирующего графика для лучшей степени (если найдено)
if best_degree:
    coefficients = np.polyfit(x_vals, y_vals, best_degree)
    polynomial = np.poly1d(coefficients)
    approximations = polynomial(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="Исходные данные", marker="o")
    plt.plot(x_vals, approximations, label=f"Аппроксимация (степень {best_degree})", linestyle="--")
    plt.title("Аппроксимация частот многочленом")
    plt.xlabel("Возраст")
    plt.ylabel("Частота")
    plt.legend()
    plt.grid()
    plt.show()
