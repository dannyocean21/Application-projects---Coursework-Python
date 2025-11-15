import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Практическая часть: простая нейронная сеть
# ------------------------------------------
# В этой части мы создадим и обучим нейронную сеть для задачи предсказания 
# цены жилья на основе площади и количества комнат.

class NeuralNetworkExample:
    def __init__(self):
        self.model = None  # Инициализируем модель
        self.x_train = None  # Входные данные для обучения
        self.y_train = None  # Целевые значения для обучения

    def prepare_data(self):
        """
        Генерирует данные для обучения.
        """
        # Данные: [площадь (м²), количество комнат]
        self.x_train = np.array([[30, 1], [50, 2], [70, 3], [90, 4], [120, 5]], dtype=np.float32)
        # Целевые значения: [цена в тысячах долларов]
        self.y_train = np.array([100, 150, 200, 250, 300], dtype=np.float32)

    def build_model(self):
        """
        Создает простую нейронную сеть с одним скрытым слоем.
        """
        # Используем последовательную модель
        self.model = Sequential([
            Dense(10, input_dim=2, activation='relu'),  # Скрытый слой с 10 нейронами
            Dense(1)  # Выходной слой
        ])
        # Компилируем модель с функцией потерь и оптимизатором
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self):
        """
        Обучает модель на подготовленных данных.
        """
        self.model.fit(self.x_train, self.y_train, epochs=100, verbose=1)

    def test_model(self):
        """
        Делает предсказания на новых данных.
        """
        # Новые данные
        x_test = np.array([[40, 1], [80, 3], [100, 4]], dtype=np.float32)
        predictions = self.model.predict(x_test)
        print("Результаты предсказаний:")
        for i, pred in enumerate(predictions):
            print(f"Данные: {x_test[i]} — Предсказанная цена: {pred[0]:.2f} тыс. долларов")

# Основной запуск программы
if __name__ == "__main__":
    nn_example = NeuralNetworkExample()
    nn_example.prepare_data()
    nn_example.build_model()
    nn_example.train_model()
    nn_example.test_model()
