import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

# Загрузка набора данных Titanic (замените 'titanic.csv' на путь к файлу)
data = pd.read_csv('titanic.csv')

# Предобработка данных
# Заполнение пропусков в столбце 'Age' средним значением
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Кодирование столбца 'Sex' (Мужчина = 0, Женщина = 1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Мы будем использовать столбцы: Pclass, Sex, Age, Siblings/Spouses Aboard, Parents/Children Aboard, Fare
# Целевая переменная — 'Survived' (1 = выжил, 0 = не выжил)

# Разделение данных на признаки (X) и целевую переменную (y)
X = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
y = data['Survived']

# Разделение данных на обучающую и тестовую выборки (80% для обучения, 20% для тестирования)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Визуализация важности признаков
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importances)
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.show()

# Кривая обучения - визуализация, как производительность модели улучшалась с увеличением объема данных
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

# Построение кривой обучения
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score", color="blue")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Cross-validation Score", color="green")
plt.title("Learning Curve")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Матрица ошибок - визуализация того, как модель классифицирует 'Survived' и 'Did not survive'
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not Survive', 'Survived'], yticklabels=['Did not Survive', 'Survived'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Кривая ROC - Receiver Operating Characteristic Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Построение ROC кривой
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="best")
plt.show()

# Кривая Precision-Recall - визуализация компромисса между точностью и полнотой
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

# Построение Precision-Recall кривой
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

# Прогноз для нового пассажира (например, Pclass = 1, Sex = 0, Age = 25, Fare = 100)
new_data = pd.DataFrame({
    'Pclass': [1],
    'Sex': [0],  # Мужчина
    'Age': [25],
    'Siblings/Spouses Aboard': [0],
    'Parents/Children Aboard': [0],
    'Fare': [100]
})

# Прогноз, выжил ли этот новый пассажир
prediction = model.predict(new_data)
print("Survived" if prediction == 1 else "Did not survive")
