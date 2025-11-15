import pandas as pd # Import library=ires pasdas for working with data
from sklearn.model_selection import train_test_split # Import functions for splitting the data
from sklearn.ensemble import RandomForestClassifier # Import the random forest
from sklearn.metrics import accuracy_score #Import metrics for defining the accuracy
import matplotlib.pyplot as plt #Import library for graphs

# Upload the data abot Titanic (change the "titanic.cv" for the path to the file)
data = pd.read_csv("titanic.csv") # Read the data from CSV file to Dataframe

# Preprocess the data
# Fill in the missing value, and use the mean value

data['Age'].fillna(data['Age'].mean(), inplace=True)
# Fillna() fills in the missing values with the mean value in the column

data["Sex"] = data["Sex"].map({"male":0, "female": 1})
# Uses the mapping and changes the male and female into numbers

# We leave only those columns which wil be used for training the model
# Observant variable - Survived (1 - lives, 0 - not lives)
X = data[["Pclass", "Sex", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"]] # regressants
y = data["Survived"] #regressor

#We need to split the data into learning and test sample(80% for learning and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_test_split splits the data into training and testing parts. 0.2 means the 20%
# random_state=42 обеспечивают воспроизводимость результатов

# Обучаем модель случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train) # Training the data

# We do the forecasting
y_pred = model.predict(X_test)
#. predict() it is used for forecasting the variable

accuracy = accuracy_score(y_test,y_pred)

print(f"Model accuracy: {accuracy*100:.2f}%")

#Построение графика
feature_importance = model.feature_importances_ #
plt.bar(X.columns, feature_importance)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.show()

new_data = pd.DataFrame({
    "Pclass" : [3],
    "Sex": [0],
    "Age": [22],
    "Siblings/Spouses Aboard": [0],
    "Parents/Children Aboard": [0],
    "Fare": [40]
})

prediction = model.predict(new_data)
print("Survived" if prediction==1 else "Did not survive")



