import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# Flatten landmarks
X = np.array([np.array(sample).flatten() for sample in data['x']])
y = np.array(data['y'])

# Strong cleaning
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0
too_large = np.abs(X) > 1e6  # Anything larger than 1 million
X[too_large] = 0.0

print(f" After strong cleaning: {X.shape[0]} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(" Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print(" Model saved to gesture_model.pkl")
