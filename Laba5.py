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
        "harmonic": ((x1 + xn) / 2, np.sqrt(y1 * yn)),
        "dependency_4": ((2 * x1 * xn) / (x1 + xn), (y1 + yn) / 2),
        "dependency_5": ((x1 + xn) / 2, (2 * y1 * yn) / (y1 + yn)),
        "dependency_6": ((2 * x1 * xn) / (x1 + xn), (2 * y1 * yn) / (y1 + yn)),
        "dependency_7": (np.sqrt(x1 * xn), (y1 + yn) / 2)
    }
    return means

# Расчет средних значений для каждой зависимости
structural_means = calculate_structural_means(data_df)

# Функция для расчета экспериментальных значений для каждой зависимости
def calculate_experimental_values(df, means):
    """Вычисляем экспериментальные значения для каждой зависимости."""
    x_values = np.array(df["Age"])
    y_values = np.array(df["Frequency"])
    experimental_values = {}

    # Используем заданные значения для зависимостей 4–7
    for dependency, (x_avg, _) in means.items():
        if dependency == "geometric":
            y_s = 796
        elif dependency == "dependency_4":
            y_s = 940.5
        elif dependency == "dependency_5":
            y_s = 926
        elif dependency == "dependency_6":
            y_s = 940.5
        elif dependency == "dependency_7":
            y_s = 796
        else:
            y_s = np.interp(x_avg, x_values, y_values)  # Интерполяция для остальных зависимостей
        experimental_values[dependency] = y_s

    return experimental_values

# Пересчитываем экспериментальные значения для каждой зависимости
experimental_values = calculate_experimental_values(data_df, structural_means)

# Расчет отклонений на основе новых экспериментальных значений
absolute_deviations = {key: abs(mean[1] - experimental_values[key]) for key, mean in structural_means.items()}
percent_deviations = {key: (dev / sum(frequencies)) * 100 for key, dev in absolute_deviations.items()}

# Вывод результатов для Задания 1
print("\nЗадание 1: Структурная идентификация (с корректными y_s)")
for key, mean in structural_means.items():
    print(f"Зависимость: {key.capitalize()}, Средние значения: ̅x_s = {round(mean[0], 1)}, ̅y_s = {round(mean[1], 1)}, "
          f"Экспериментальное значение y_s = {round(experimental_values[key], 1)}, Отклонение Δ_s = {round(absolute_deviations[key], 1)} "
          f"({round(percent_deviations[key], 1)}%)")

min_deviation_key = min(absolute_deviations, key=absolute_deviations.get)
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
    coefficients = np.polyfit(x_vals, y_vals, degree)
    polynomial = np.poly1d(coefficients)

    approximations = polynomial(x_vals)
    differences = np.abs(y_vals - approximations)

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
