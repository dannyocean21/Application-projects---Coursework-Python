import torch
import torch.nn as nn #NEURAL NETWORK
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. Подготовка данных: Текстовые предложения и их метки (1 - позитивный, 0 - негативный)
# Это наш обучающий набор данных, где каждое предложение имеет соответствующую метку.
data = [
    ("I love this product", 1),  # Позитивное
    ("This is the worst thing I've ever bought", 0),  # Негативное
    ("Amazing quality, highly recommend it", 1),  # Позитивное
    ("Not worth the money, really bad", 0),  # Негативное
    ("Great value for the price", 1),  # Позитивное
    ("Terrible experience, do not buy", 0),  # Негативное
    ("The glass is half full", 1),
    ("The glass is half empty", 0),
    ("I was disappointed by such bad quality",0),
    ("Everyday life is perishible", 0),
    ("You cook the best pasta I have ever eaten", 1),
    ("Great product! Very happy with the purchase.", 1),
    ("This is the worst thing I've ever bought", 0),
    ("Every your day is worth a million bucks", 1),
    ("I have nothing but my sorrow", 0),
    ("Today i don't feel like doing anything", 0),
    ("I'm stuck in traffic for 40 minutes already",0),
    ("The worst day of my life", 0),
    ("Have some reaaly nice girl", 1),
    ("He made her feel special", 1),
    ("I live my best life", 1),
    ("The end of the workday",1)
]

# 2. Создаем словарь с токеном <UNK> для обработки неизвестных слов
# <UNK> (unknown) используется для слов, которых нет в словаре.
word_to_index = {"<UNK>": 0}

# Индексируем слова, создавая отображение слово -> индекс
index = 1
for sentence, _ in data:  # Проходим по всем предложениям из набора данных
    for word in sentence.split():  # Разбиваем предложение на слова
        if word not in word_to_index:  # Если слово еще не в словаре
            word_to_index[word] = index  # Добавляем его с уникальным индексом
            index += 1  # Увеличиваем индекс для следующего слова

# 3. Функция для преобразования текста в последовательность индексов
# Если слово отсутствует в словаре, возвращается индекс <UNK>
def encode_sentence(sentence):
    return [word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence.split()]

# Кодируем данные: Преобразуем текст в индексы и метки в тензоры
encoded_data = [(torch.tensor(encode_sentence(sentence)), label) for sentence, label in data]

# 4. Определяем Dataset для работы с PyTorch
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Сохраняем данные в классе

    def __len__(self):
        return len(self.data)  # Возвращаем длину набора данных

    def __getitem__(self, idx):
        return self.data[idx]  # Возвращаем элемент по индексу

# Создаем экземпляр датасета
dataset = TextDataset(encoded_data)

# 5. Функция для подготовки батчей (Padding предложений до одинаковой длины)
def collate_fn(batch):
    # Разделяем батч на предложения и метки
    sentences, labels = zip(*batch)
    # Выравниваем длину предложений путем добавления паддинга (значение 0)
    batch_padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    # Преобразуем метки в тензор
    return batch_padded_sentences, torch.tensor(labels)

# Создаем DataLoader для работы с батчами данных
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 6. Определяем модель классификации
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SentimentClassifier, self).__init__()
        # Embedding слой: Преобразует индексы слов в векторы заданной размерности
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Полносвязный слой: Классифицирует предложение как позитивное или негативное
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # Преобразуем индексы слов в векторы
        x = self.embedding(x)
        # Усредняем векторы слов (mean pooling)
        x = x.mean(dim=1)
        # Пропускаем через полносвязный слой
        x = self.fc(x)
        # Применяем сигмоид для получения вероятности
        return torch.sigmoid(x)

# Инициализируем модель
vocab_size = len(word_to_index)  # Размер словаря (число уникальных слов)
embed_dim = 15 # Размерность векторов слов (embedding)
model = SentimentClassifier(vocab_size, embed_dim)

# 7. Настраиваем функцию потерь и оптимизатор
criterion = nn.BCELoss()  # Бинарная кросс-энтропия, используемая для задачи классификации
optimizer = optim.Adam(model.parameters(), lr=0.05)  # Оптимизатор Adam RELU

# !LEARNING RATE

#! 8. Обучение модели
print("Начало обучения...")
for epoch in range(25):  # Количество эпох
    for sentences, labels in dataloader:
        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(sentences)  # Пропускаем данные через модель
        # Считаем потери между предсказанием и реальными метками
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()  # Вычисляем градиенты
        optimizer.step()  # Обновляем параметры модели

    # Выводим потери для каждой эпохи
    print(f"Эпоха {epoch + 1}, Потери: {loss.item():.4f}")

# 9. Тестирование модели
test_sentence = "I don't feel like doing anything today."  # Новое предложение для тестирования
encoded_test = torch.tensor(encode_sentence(test_sentence))  # Кодируем предложение
padded_test = pad_sequence([encoded_test], batch_first=True, padding_value=0)  # Применяем паддинг
prediction = model(padded_test).item()  # Получаем предсказание
# Выводим результат
print(f"Предсказание для '{test_sentence}': {'Позитивное' if prediction > 0.5 else 'Негативное'}")
