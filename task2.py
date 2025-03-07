import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap


df = pd.read_csv('data.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# Define a dictionary to map label numbers to clothing types
clothing_labels = {
    0: "T-shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
}

# Display Sample Images with Labels
plt.figure(figsize=(10, 10))
for label in range(10):
    images = df[df['label'] == label].iloc[:5, 1:].values
    for i in range(5):
        plt.subplot(10, 5, label * 5 + i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(clothing_labels[label])
plt.tight_layout()
plt.show()

# Data Preprocessing
X = df.drop('label', axis=1).values / 255.0  # Normalize pixel values
y = df['label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(clothing_labels.values()), yticklabels=list(clothing_labels.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Explainable AI with SHAP
explainer = shap.LinearExplainer(model, X_train, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test[:5])
shap.image_plot(shap_values, X_test[:5].reshape(-1, 28, 28))