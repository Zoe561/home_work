import csv
import math
import random
import os
import numpy as np

# 標準化函數
def standardize(data):
    mean = sum(data) / len(data)
    std = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
    std = max(std, 1e-12)  # 避免除以零
    return [(x - mean) / std for x in data], mean, std

# 加載數據
class WeightHeightDataset:
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳過標頭
            for row in reader:
                if len(row) < 3:
                    continue
                gender_str = row[0].strip()
                height = float(row[1])
                weight = float(row[2])
                gender_val = 1.0 if gender_str.lower().startswith('m') else 0.0
                self.data.append((gender_val, height, weight))
        
        # 進行標準化
        heights, self.mean_h, self.std_h = standardize([d[1] for d in self.data])
        weights, self.mean_w, self.std_w = standardize([d[2] for d in self.data])
        self.features = [(d[0], heights[i]) for i, d in enumerate(self.data)]
        self.labels = weights
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return np.array(self.features[index], dtype=np.float32), np.array(self.labels[index], dtype=np.float32)

# 簡單的神經網路
class RegressionNN:
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        self.weights_h = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_h = np.zeros((1, hidden_size))
        self.weights_o = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_o = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        self.z_hidden = np.dot(x, self.weights_h) + self.bias_h
        self.a_hidden = self.relu(self.z_hidden)
        self.z_output = np.dot(self.a_hidden, self.weights_o) + self.bias_o
        return self.z_output
    
    def backward(self, x, y, output, learning_rate=0.01):
        m = x.shape[0]
        d_loss = 2 * (output - y.reshape(-1, 1)) / m
        
        d_weights_o = np.dot(self.a_hidden.T, d_loss)
        d_bias_o = np.sum(d_loss, axis=0, keepdims=True)
        
        d_hidden = np.dot(d_loss, self.weights_o.T) * self.relu_derivative(self.z_hidden)
        d_weights_h = np.dot(x.T, d_hidden)
        d_bias_h = np.sum(d_hidden, axis=0, keepdims=True)
        
        self.weights_h -= learning_rate * d_weights_h
        self.bias_h -= learning_rate * d_bias_h
        self.weights_o -= learning_rate * d_weights_o
        self.bias_o -= learning_rate * d_bias_o

# 訓練模型
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "gender-height-weight.csv")
dataset = WeightHeightDataset(csv_path)
data_size = len(dataset)
train_size = int(0.8 * data_size)
test_size = data_size - train_size
indices = list(range(data_size))
random.shuffle(indices)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_data = [dataset[i] for i in train_indices]
test_data = [dataset[i] for i in test_indices]

X_train, y_train = np.array([d[0] for d in train_data]), np.array([d[1] for d in train_data])
X_test, y_test = np.array([d[0] for d in test_data]), np.array([d[1] for d in test_data])

model = RegressionNN()
epochs = 2000
learning_rate = 0.001

for epoch in range(epochs):
    predictions = model.forward(X_train)
    model.backward(X_train, y_train, predictions, learning_rate)
    loss = np.mean((predictions - y_train.reshape(-1, 1)) ** 2)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# 評估模型
predictions = model.forward(X_test)
predictions = predictions * dataset.std_w + dataset.mean_w
y_test = y_test * dataset.std_w + dataset.mean_w
total_error = np.sum(np.abs(predictions - y_test.reshape(-1, 1)))
count = len(y_test)

print(f"[Test] 平均誤差 (磅): {total_error / count:.3f}")
