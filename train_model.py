import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# Flatten landmarks
X = np.array([np.array(sample).flatten() for sample in data['x']])
y = np.array(data['y'])

# normalize transformed landmarks
# minmax = MinMaxScaler()
# X = minmax.fit_transform(X)

print(f" Before strong cleaning: {y.shape[0]} samples")
# Apply Min-Max Scaling per sample
def minmax_scale(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

X_scaled = np.array([minmax_scale(sample) for sample in X])
print("[DEBUG] First scaled sample (min/max):", np.min(X_scaled[0]), "/", np.max(X_scaled[0]))
print("[DEBUG] Sample values:", X_scaled[0][:5], "...")



# Strong cleaning
# X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0
# too_large = np.abs(X) > 1e6  # Anything larger than 1 million
# X[too_large] = 0.0

print(f" After strong cleaning: {X.shape[0]} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
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
