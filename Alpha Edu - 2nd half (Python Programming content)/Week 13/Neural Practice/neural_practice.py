# Импортируем необходимые библиотеки
import torch
import torch.nn as nn  # Библиотека для работы с нейронными сетями
import torch.optim as optim  # Оптимизаторы для настройки параметров модели
from torchvision import datasets, transforms  # Работа с набором данных MNIST
from torch.utils.data import DataLoader  # Загрузчики данных
import matplotlib.pyplot as plt  # Для визуализации данных

# *** ШАГ 1: Загрузка и подготовка данных ***
# Преобразования для нормализации данных
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразуем изображения в тензоры PyTorch
    transforms.Normalize((0.5,), (0.5,))  # Нормализация (центрирование данных)
])

# Загрузка тренировочного и тестового набора данных MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Создаем загрузчики данных
#!batch_size=64, 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Для обучения
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Для тестирования

# *** ШАГ 2: Определяем архитектуру нейронной сети ***
class SimpleNN(nn.Module):
    """
    Простая полносвязная нейронная сеть с двумя слоями.
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Входной слой: 28x28 -> 128 узлов
        self.fc2 = nn.Linear(128, 64)  # Скрытый слой: 128 -> 64 узлов
        self.fc3 = nn.Linear(64, 10)  # Выходной слой: 64 -> 10 узлов (классы)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Преобразуем изображение (28x28) в вектор 
        x = torch.relu(self.fc1(x))  # Активация ReLU на первом слое
        x = torch.relu(self.fc2(x))  # Активация ReLU на втором слое
        x = self.fc3(x)  # Линейный выход (без активации) для классов
        return x

# Создаем объект модели
model = SimpleNN()

# *** ШАГ 3: Настройка функции потерь и оптимизатора ***
criterion = nn.CrossEntropyLoss()  # Кросс-энтропия для задач классификации
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Оптимизатор SGD (градиентный спуск)

# *** ШАГ 4: Обучение модели ***
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """
    Функция для обучения модели.
    """
    for epoch in range(epochs):
        model.train()  # Устанавливаем режим обучения
        running_loss = 0.0  # Переменная для накопления ошибки
        for images, labels in train_loader:
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(images)  # Прямой проход
            loss = criterion(outputs, labels)  # Вычисляем ошибку
            loss.backward()  # Обратное распространение ошибки
            optimizer.step()  # Обновляем параметры
            running_loss += loss.item()
        print(f"Эпоха {epoch + 1}/{epochs}, Потеря: {running_loss / len(train_loader):.4f}")

# Обучаем модель
train_model(model, train_loader, criterion, optimizer, epochs=5)

# *** ШАГ 5: Оценка модели на тестовых данных ***
def evaluate_model(model, test_loader):
    """
    Оценивает точность модели на тестовом наборе данных.
    """
    model.eval()  # Переключаем модель в режим оценки
    correct = 0
    total = 0
    with torch.no_grad():  # Отключаем градиенты
        for images, labels in test_loader:
            outputs = model(images)  # Прогоняем тестовые данные через модель
            _, predicted = torch.max(outputs, 1)  # Находим предсказанный класс
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Точность модели: {100 * correct / total:.2f}%")

# Оцениваем модель
evaluate_model(model, test_loader)

# *** ШАГ 6: Визуализация предсказаний ***
def visualize_predictions(model, test_loader):
    """
    Визуализирует первые несколько изображений из тестового набора с предсказаниями.
    """
    model.eval()
    images, labels = next(iter(test_loader))  # Берем первую партию данных
    outputs = model(images)  # Получаем предсказания
    _, predicted = torch.max(outputs, 1)  # Извлекаем предсказанные метки

    # Отображаем 6 изображений с их предсказаниями
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].view(28, 28), cmap="gray")  # Визуализируем изображение
        plt.title(f"Истинное: {labels[i]}\nПредсказание: {predicted[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Визуализируем предсказания
visualize_predictions(model, test_loader)
