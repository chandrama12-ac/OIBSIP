import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import datasets

# Load original iris dataset from sklearn
iris = datasets.load_iris()

# Create custom DataFrame with 'id' and clean column names
df = pd.DataFrame(iris.data, columns=[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
])
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df['id'] = df.index + 1  # unique id starting from 1

# Reorder columns
df = df[['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']]

# Preview the data
print("Sample of the dataset:")
print(df.head())

# ---- Data Visualization ----

# Pairplot
sns.pairplot(df.drop('id', axis=1), hue='species')
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.drop(['id', 'species'], axis=1).corr(), annot=True, cmap='Purples')
plt.title("Feature Correlation Heatmap")
plt.show()

# ---- Model Training ----

# Features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---- Sample Prediction ----

sample = [[5.0, 3.4, 1.5, 0.2]]  # likely Setosa
prediction = model.predict(sample)
print(f"\nPrediction for sample {sample}: {prediction[0]}")
