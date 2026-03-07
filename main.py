import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # to save the model

# 1. Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# 2. Prepare features and target
X = data.drop('label', axis=1)
y = data['label']

# Encode crop names as numbers
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. Test model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# 6. Save trained model and label encoder for GUI
joblib.dump(model, "model/decision_tree.pkl")
joblib.dump(le, "model/label_encoder.pkl")
print("Model and label encoder saved in 'model/' folder.")







