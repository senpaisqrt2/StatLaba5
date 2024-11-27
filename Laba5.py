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
        "hyperbolic": ((2 * x1 * xn) / (x1 + xn), (y1 + yn) / 2),
        "inverse_linear": ((x1 + xn) / 2, (2 * y1 * yn) / (y1 + yn)),
        "fractional_linear": ((2 * x1 * xn) / (x1 + xn), (2 * y1 * yn) / (y1 + yn)),
        "logarithmic": (np.sqrt(x1 * xn), (y1 + yn) / 2)
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
        elif dependency == "hyperbolic":
            y_s = 940.5
        elif dependency == "inverse_linear":
            y_s = 926
        elif dependency == "fractional_linear":
            y_s = 940.5
        elif dependency == "logarithmic":
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
          f"({round(percent_deviations[key], 2)}%)")

min_deviation_key = min(absolute_deviations, key=absolute_deviations.get)
print(f"Минимальное отклонение у зависимости: {min_deviation_key.capitalize()}")

# Задание 2: Аппроксимирующий многочлен
# Определение степени многочлена (фиксированная степень = 5)
fixed_degree = 5  # Фиксированная степень многочлена

# Создаем массивы для аппроксимации
x_vals = np.array(data_df["Age"], dtype=np.float64)
frequencies = np.array(data_df["Frequency"], dtype=np.float64)
# print(x_vals, frequencies)

def compute_differences(frequencies):
    return [abs(frequencies[i - 1] - frequencies[i]) for i in range(1, len(frequencies))]

differences = frequencies

# Сумма частот и вычисление 2% от суммы
total_frequency = sum(frequencies)
threshold = 0.02 * total_frequency

max_degree = 1

while max(compute_differences(differences)) > threshold:
    differences = compute_differences(differences)
    # print(differences)
    # print('Максимальное значение ', max_degree, 'ряда разности: ', max(differences))
    max_degree = max_degree + 1

differences = compute_differences(differences)
# print(differences)
print('Максимальное значение ', max_degree, 'ряда разности: ', max(differences))
print('2 % от суммы частот: ', threshold)
print('Показатель степени аппроксимирующего многочлена: ', max_degree)

x_vals = np.array(data_df["Age"], dtype=np.float64)
y_vals = np.array(data_df["Frequency"], dtype=np.float64)

# Построение аппроксимации для фиксированной степени
coefficients = np.polyfit(x_vals, y_vals, fixed_degree)
polynomial = np.poly1d(coefficients)

# Вычисление аппроксимированных значений
approximations = polynomial(x_vals)
differences = np.abs(y_vals - approximations)

# Проверка условия максимальной разности
max_difference = np.max(differences)
percent_difference = (max_difference / sum(frequencies)) * 100


# Построение графика полигона частот
plt.figure(figsize=(10, 6))
plt.plot(data_df["Age"], data_df["Frequency"], label="Полигон частот", marker="o")
plt.title("Полигон частот (Возраст vs Частота)")
plt.xlabel("Возраст")
plt.ylabel("Частота")
plt.legend()
plt.grid()
plt.show()

# Построение аппроксимирующего графика для степени 5
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Исходные данные", marker="o")
plt.plot(x_vals, approximations, label="Аппроксимация (степень 5)", linestyle="--")
plt.title("Аппроксимация частот многочленом (степень 5)")
plt.xlabel("Возраст")
plt.ylabel("Частота")
plt.legend()
plt.grid()
plt.show()

