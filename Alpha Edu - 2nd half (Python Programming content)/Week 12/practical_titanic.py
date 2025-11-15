import pandas as pd  # Импорт библиотеки pandas для работы с данными
from sklearn.model_selection import train_test_split  # Импорт функции для разделения данных на обучающую и тестовую выборки
from sklearn.ensemble import RandomForestClassifier  # Импорт алгоритма случайного леса (RandomForest) для классификации
from sklearn.metrics import accuracy_score, confusion_matrix  # Импорт метрики для оценки точности модели
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков
import seaborn as sns

# Загрузка данных о Титанике (замените 'titanic.csv' на путь к вашему файлу)
data = pd.read_csv('titanic.csv')  # Чтение данных из CSV файла в DataFrame

# Предобработка данных
# Заполнение пропущенных значений в столбце 'Age' средним значением по этому столбцу
data['Age'].fillna(data['Age'].mean(), inplace=True)  
# .fillna() заполняет все пропущенные (NaN) значения в столбце 'Age' средним значением из этого столбца

# Кодирование столбца 'Sex' (мужчина = 0, женщина = 1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  
# .map() используется для преобразования категориальных значений в числовые: 'male' становится 0, 'female' — 1

# Мы оставляем только те столбцы, которые будем использовать для обучения модели
# Целевая переменная — 'Survived' (1 = выжил, 0 = не выжил)
X = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]  # Признаки
y = data['Survived']  # Целевая переменная

# Разделяем данные на обучающую и тестовую выборки (80% для обучения, 20% для тестирования)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_test_split разделяет данные на обучающую и тестовую выборки. Параметр test_size=0.2 указывает, что 20% данных идут в тестовую выборку
# random_state=42 обеспечивает воспроизводимость результатов

# Обучаем модель случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)  
# RandomForestClassifier — это алгоритм случайного леса для классификации. Параметр n_estimators=100 указывает количество деревьев в лесу
# random_state=42 задает начальное состояние генератора случайных чисел для воспроизводимости

model.fit(X_train, y_train)  # Обучение модели на обучающих данных

# Делаем предсказания на тестовой выборке
y_pred = model.predict(X_test)  
# .predict() используется для предсказания целевой переменной (Survived) на основе тестовых данных

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)  
# accuracy_score вычисляет точность модели, сравнив предсказания (y_pred) с реальными значениями (y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Вывод точности модели в процентах


# #* Построение графика важности признаков
# feature_importances = model.feature_importances_  # Получаем важность признаков, определённую моделью случайного леса

# plt.bar(X.columns, feature_importances)  # Строим столбчатую диаграмму важности признаков
# plt.title("Feature Importances")  # Заголовок графика
# plt.ylabel("Importance")  # Подпись оси Y
# plt.xlabel("Feature")  # Подпись оси X
# plt.xticks(rotation=45)  # Поворот подписей на оси X для лучшей читаемости
# plt.show()  # Отображение графика



# # Прогнозирование для нового пассажира (например, Pclass = 1, Sex = 0, Age = 25, Fare = 100)
# new_data = pd.DataFrame ({
#     'Pclass': [1],
#     'Sex': [0],
#     'Age': [12],
#     'Siblings/Spouses Aboard': [3],
#     'Parents/Children Aboard': [1],
#     'Fare': [140]
# })

# # Прогнозируем, выжил ли этот пассажир
# prediction = model.predict(new_data)  # Прогнозирование для новых данных
# print("Survived" if prediction == 1 else "Did not survive")  # Вывод результата прогноза (выжил или не выжил)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not Survive', 'Survived'], yticklabels=['Did not Survive', 'Survived'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()