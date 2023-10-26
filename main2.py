import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle

# Load the dataset (Make sure to put the data file in the current directory)
data = pd.read_csv("./data.csv")

# Data Exploration
print(data.info())
print('-' * 30)
print(data.describe())
print('-' * 30)
print(data.describe(include=['O']))
print('-' * 30)
print(data.head())
print('-' * 30)
print(data.tail())

# Data Cleaning
data.drop("id", axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Visualize the tumor diagnosis results
sns.countplot(data['diagnosis'], label="Count")
plt.show()

# Visualize the correlation between mean features using a heatmap
corr = data[data.columns[1:11]].corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True)
plt.show()

# Feature Selection
features_remain = data.columns[1:31]
print(features_remain)
print('-' * 100)

# Split the data into training and testing sets (70% training, 30% testing)
train, test = train_test_split(data, test_size=0.3)
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']

# Normalize the data using Z-Score to ensure zero mean and unit variance
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# Create an SVM classifier
model = SVC()
# Train the model on the training set
model.fit(train_X, train_y)
# Make predictions on the test set
predictions = model.predict(test_X)
accuracy = accuracy_score(predictions, test_y)
print('SVM Accuracy: {:.4f}'.format(accuracy))

# Data Visualization (Example: Age Distribution)
plt.figure(figsize=(8, 6))
sns.histplot(data['radius_mean'], bins=20, kde=True, color='skyblue')
plt.title('Radius Mean Distribution')
plt.xlabel('Radius Mean')
plt.ylabel('Count')
plt.show()

# Deep Learning with TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Input(shape=(train_X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_X, train_y, epochs=10, batch_size=32, validation_split=0.2)

# Deep Learning with PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init()
        self.fc1 = nn.Linear(train_X.shape[1], 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_X, train_y = shuffle(train_X, train_y)
train_X_tensor = torch.FloatTensor(train_X)
train_y_tensor = torch.FloatTensor(train_y.to_numpy()).view(-1, 1)

test_X_tensor = torch.FloatTensor(test_X)

train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / (len(train_loader)):.4f}")

# Evaluate the PyTorch model
model.eval()
with torch.no_grad():
    outputs = model(test_X_tensor)
    predicted = (outputs > 0.5).float()
    predicted = predicted.numpy().flatten()
    
# Convert predictions to integers (0 or 1)
predicted = predicted.astype(int)

# Output the predictions
test['diagnosis'] = predicted
test[['id', 'diagnosis']].to_csv('breast_cancer_predictions.csv', index=False)
