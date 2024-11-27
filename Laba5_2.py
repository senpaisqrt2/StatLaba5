import collections
import numpy as np

# Загрузка данных из файла
with open('Статистика.txt', 'r') as file:
    data = list(map(int, file.readlines()))

# Получаем дискретный ряд распределения
distribution = collections.Counter(data)
sorted_distribution = dict(sorted(distribution.items()))

# Извлекаем значения возраста и частоты
ages = list(sorted_distribution.keys())
frequencies = list(sorted_distribution.values())


# Вычисление ряда разностей
def compute_differences(frequencies):
    return [abs(frequencies[i - 1] - frequencies[i]) for i in range(1, len(frequencies))]


differences = frequencies
# differences2 = compute_differences(differences1)
# differences3 = compute_differences(differences2)
# differences4 = compute_differences(differences3)

# print(differences1)
# print(differences2)
# print(differences3)
# print(differences4)

# Сумма частот и вычисление 2% от суммы
total_frequency = sum(frequencies)
threshold = 0.02 * total_frequency

max_degree = 1

while max(compute_differences(differences)) > threshold:
    differences = compute_differences(differences)
    print(differences)
    print('Максимальное значение ', max_degree, 'ряда разности: ', max(differences))
    max_degree = max_degree + 1

differences = compute_differences(differences)
print(differences)
print('Максимальное значение ', max_degree, 'ряда разности: ', max(differences))

print('2 % от суммы частот: ', threshold)

print('Показатель степени аппроксимирующего многочлена: ', max_degree)

# print(max_degree)
# Определение максимального значения разностей
# max_difference1 = max(differences1)
# max_difference2 = max(differences2)
# max_difference3 = max(differences3)
# max_difference4 = max(differences4)

# max_degree = None

# print(max_difference1, ' ', max_difference2 , ' ', threshold)
# print (max_difference3)
# print (max_difference4)
