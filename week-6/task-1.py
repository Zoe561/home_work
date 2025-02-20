import csv
import math
import random
import os
from network import Network, relu  # 引用 network.py

def load_weight_height_data(csv_file):
    data = []
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
            data.append((gender_val, height, weight))
    return data

def standardize(data_list, idx):
    vals = [d[idx] for d in data_list]
    mean_val = sum(vals) / len(vals)
    std_val = math.sqrt(sum((v - mean_val)**2 for v in vals) / len(vals))

    std_val = max(std_val, 1e-12)  # 避免分母為 0
    for i in range(len(data_list)):
        tmp = list(data_list[i])
        tmp[idx] = (tmp[idx] - mean_val) / std_val
        data_list[i] = tuple(tmp)
    return mean_val, std_val

def run_regression_task():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    height_weight_path = os.path.join(script_dir, "gender-height-weight.csv")
    dataset = load_weight_height_data(height_weight_path)

    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data, test_data = dataset[:split_idx], dataset[split_idx:]

    mean_h, std_h = standardize(train_data, 1)
    mean_w, std_w = standardize(train_data, 2)

    for i in range(len(test_data)):
        g, h, w = test_data[i]
        test_data[i] = (g, (h - mean_h) / std_h, (w - mean_w) / std_w)

    nn = Network(input_size=2, hidden_size=2, output_size=1, hidden_activation=relu, output_activation='linear')

    epochs, lr = 5, 0.01
    for ep in range(epochs):
        for g, h, w in train_data:
            nn.forward_train(g, h)
            nn.backward([w], loss_type='mse')
            nn.zero_grad(learning_rate=lr)

        train_loss_sum = sum((w - nn.forward(g, h)[0])**2 for g, h, w in train_data)
        print(f"Epoch {ep+1}/{epochs}, Train MSE: {train_loss_sum / len(train_data):.4f}")

    err_sum = sum(abs((nn.forward(g, h)[0] * std_w + mean_w) - (w * std_w + mean_w)) for g, h, w in test_data)
    print(f"[Test] Average Error in Pounds: {err_sum / len(test_data):.3f}")

if __name__ == "__main__":
    print("=== [Task-1] Regression: Predict Weight by Gender & Height ===")
    run_regression_task()
