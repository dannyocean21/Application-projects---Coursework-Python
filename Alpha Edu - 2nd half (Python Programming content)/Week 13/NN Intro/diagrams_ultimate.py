import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # Добавляем seaborn для создания тепловых карт

# Класс для создания различных графиков с помощью matplotlib и seaborn для данных Titanic.
class GraphBuilder:

    def __init__(self):
        # Инициализация класса
        pass

    def survival_rate_by_gender(self, file_path):
        """
        Визуализирует уровень выживаемости по полу из файла Titanic.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)
        survival_by_gender = data.groupby("Sex")["Survived"].mean()

        # Барчарт выживаемости по полу
        plt.figure(figsize=(10, 6))
        plt.bar(survival_by_gender.index, survival_by_gender.values, color=['blue', 'pink'])
        plt.title("Survival Rate by Gender")  # Устанавливаем заголовок графика
        plt.xlabel("Gender")  # Подпись оси X
        plt.ylabel("Survival Rate")  # Подпись оси Y
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Добавляем сетку
        plt.show()

    def age_distribution(self, file_path):
        """
        Показывает распределение возраста пассажиров.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)
        ages = data["Age"].dropna()

        # Гистограмма распределения возраста
        plt.figure(figsize=(10, 6))
        plt.hist(ages, bins=20, color='green', alpha=0.7)
        plt.title("Age Distribution")  # Устанавливаем заголовок графика
        plt.xlabel("Age")  # Подпись оси X
        plt.ylabel("Number of Passengers")  # Подпись оси Y
        plt.grid(True)  # Добавляем сетку
        plt.show()

    def survival_rate_by_class(self, file_path):
        """
        Показывает уровень выживаемости по классам.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)
        survival_by_class = data.groupby("Pclass")["Survived"].mean()

        # Барчарт выживаемости по классам
        plt.figure(figsize=(10, 6))
        plt.bar(survival_by_class.index, survival_by_class.values, color='skyblue')
        plt.title("Survival Rate by Passenger Class")  # Устанавливаем заголовок графика
        plt.xlabel("Passenger Class")  # Подпись оси X
        plt.ylabel("Survival Rate")  # Подпись оси Y
        plt.xticks([1, 2, 3], labels=["First Class", "Second Class", "Third Class"])  # Названия категорий оси X
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Добавляем сетку
        plt.show()

    def fare_by_class(self, file_path):
        """
        Показывает среднюю стоимость билета по классам.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)
        fare_by_class = data.groupby("Pclass")["Fare"].mean()

        # Boxplot стоимости билетов по классам
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Pclass", y="Fare", data=data, palette="pastel")  # Создаем boxplot для анализа стоимости
        plt.title("Fare Distribution by Passenger Class")  # Устанавливаем заголовок графика
        plt.xlabel("Passenger Class")  # Подпись оси X
        plt.ylabel("Fare")  # Подпись оси Y
        plt.show()

    def survival_by_age_group(self, file_path):
        """
        Показывает уровень выживаемости по возрастным группам.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)
        data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 80], labels=['Child', 'Young Adult', 'Middle Age', 'Senior'])
        survival_by_age_group = data.groupby("AgeGroup")["Survived"].mean()

        # Барчарт выживаемости по возрастным группам
        plt.figure(figsize=(10, 6))
        plt.bar(survival_by_age_group.index, survival_by_age_group.values, color='purple')
        plt.title("Survival Rate by Age Group")  # Устанавливаем заголовок графика
        plt.xlabel("Age Group")  # Подпись оси X
        plt.ylabel("Survival Rate")  # Подпись оси Y
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Добавляем сетку
        plt.show()

    def survival_heatmap(self, file_path):
        """
        Создает тепловую карту для анализа выживаемости.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)
        heatmap_data = data.pivot_table(index='Pclass', columns='Sex', values='Survived', aggfunc='mean')

        # Тепловая карта с помощью seaborn heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)  # Создаем тепловую карту
        plt.title("Survival Heatmap by Class and Gender")  # Устанавливаем заголовок графика
        plt.xlabel("Gender")  # Подпись оси X
        plt.ylabel("Passenger Class")  # Подпись оси Y
        plt.show()

    def class_gender_survival_countplot(self, file_path):
        """
        Показывает количество выживших по классам и полу.
        :param file_path: Путь к CSV-файлу
        """
        data = pd.read_csv(file_path)

        # Countplot для анализа выживаемости по полу и классам
        plt.figure(figsize=(12, 6))
        sns.countplot(x="Pclass", hue="Sex", data=data, palette="coolwarm")
        plt.title("Count of Passengers by Class and Gender")  # Устанавливаем заголовок графика
        plt.xlabel("Passenger Class")  # Подпись оси X
        plt.ylabel("Count")  # Подпись оси Y
        plt.legend(title="Gender")  # Добавляем легенду
        plt.show()

# Пример использования
if __name__ == "__main__":
    builder = GraphBuilder()

    # Укажите путь к CSV-файлу Titanic
    titanic_csv_path = "titanic.csv"

    # Визуализация выживаемости по полу
    builder.survival_rate_by_gender(titanic_csv_path)

    # Визуализация распределения возраста
    builder.age_distribution(titanic_csv_path)

    # Визуализация уровня выживаемости по классам
    builder.survival_rate_by_class(titanic_csv_path)

    # Визуализация стоимости билетов по классам с помощью boxplot
    builder.fare_by_class(titanic_csv_path)

    # Визуализация выживаемости по возрастным группам
    builder.survival_by_age_group(titanic_csv_path)

    # Визуализация тепловой карты выживаемости
    builder.survival_heatmap(titanic_csv_path)

    # Визуализация количества пассажиров по классам и полу
    builder.class_gender_survival_countplot(titanic_csv_path)
