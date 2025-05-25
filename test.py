import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array([np.array(sample).flatten() for sample in data['x']])
y = np.array(data['y'])

print(X.max())
print(X)

# encoder = OneHotEncoder()
# y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# print(np.unique(y_encoded))