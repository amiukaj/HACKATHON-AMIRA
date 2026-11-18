import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = list(data_dict['data'])
labels = list(data_dict['labels'])


counts = Counter(labels)
for i in range(len(labels)):
    if counts[labels[i]] < 2:
        data.append(data[i])
        labels.append(labels[i])

data = np.asarray(data)
labels = np.asarray(labels)

print(f"Loaded {len(data)} total samples across {len(set(labels))} letters.")


x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved successfully to model.p!")


