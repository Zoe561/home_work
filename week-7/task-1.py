import csv
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============
#  資料前處理
# ==============

def load_data(csv_file):
    """
    讀取性別、身高與體重資料，回傳列表 [(gender_val, height, weight), ...]
    gender_val: 男=1.0, 女=0.0
    """
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

def compute_mean_std(values):
    """計算一維資料的平均值與標準差（含極小值保護）"""
    mean_ = sum(values) / len(values)
    var_ = sum((x - mean_)**2 for x in values) / len(values)
    std_ = math.sqrt(var_)
    std_ = max(std_, 1e-7)
    return mean_, std_

# =======================
#  自定義 PyTorch Dataset
# =======================
class WeightHeightDataset(Dataset):
    """
    將 gender, height 做為特徵，weight 做為標籤。
    會在 __init__ 傳入 train_mean_h, train_std_h, train_mean_w, train_std_w
    以便正確做標準化。
    """
    def __init__(self, data, mean_h, std_h, mean_w, std_w, transform=True):
        """
        data: [(gender_val, height, weight), ...]
        mean_h, std_h, mean_w, std_w: 用於標準化
        transform: 是否要做標準化 (train/test 同參數)
        """
        self.transform = transform
        self.mean_h = mean_h
        self.std_h = std_h
        self.mean_w = mean_w
        self.std_w = std_w

        self.features = []
        self.labels = []
        for (g, h, w) in data:
            if transform:
                # 只針對 height, weight 做標準化
                h_std = (h - mean_h) / std_h
                w_std = (w - mean_w) / std_w
                self.features.append([g, h_std])
                self.labels.append(w_std)
            else:
                # 如果不做標準化，就直接用原值
                self.features.append([g, h])
                self.labels.append(w)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

# =======================
#   定義神經網路模型
# =======================
class RegressionModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 1. 取得檔案路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "gender-height-weight.csv")

    # 2. 讀取資料
    all_data = load_data(csv_path)
    data_size = len(all_data)
    print(f"Total data size: {data_size}")

    # 3. Train/Test Split (8:2)
    random.shuffle(all_data)
    train_size = int(0.8 * data_size)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    # 4. 用訓練集計算 height, weight 的 mean/std
    train_heights = [d[1] for d in train_data]
    train_weights = [d[2] for d in train_data]
    mean_h, std_h = compute_mean_std(train_heights)
    mean_w, std_w = compute_mean_std(train_weights)

    # 5. 分別建立 Dataset (用相同的 mean, std)
    train_dataset = WeightHeightDataset(train_data, mean_h, std_h, mean_w, std_w, transform=True)
    test_dataset  = WeightHeightDataset(test_data,  mean_h, std_h, mean_w, std_w, transform=True)

    # 6. 建立 DataLoader (mini-batch)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 7. 建立模型
    model = RegressionModel(input_dim=2, hidden_dim=10, output_dim=1)
    # 8. 定義損失函數與優化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 9. 開始訓練
    epochs = 200
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            # x_batch: [batch_size, 2]
            # y_batch: [batch_size]

            # forward
            preds = model(x_batch)  # [batch_size, 1]
            preds = preds.squeeze(1)  # [batch_size]

            loss = criterion(preds, y_batch)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(train_dataset)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 10. 測試階段 (評估平均絕對誤差)
    model.eval()
    total_error = 0.0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch).squeeze(1)  # [batch_size]

            # 將預測結果 / 標籤 從標準化後的值 => 還原到實際體重
            preds_origin = preds * std_w + mean_w
            y_origin = y_batch * std_w + mean_w

            # 計算這個 batch 的絕對誤差總和
            abs_error = torch.abs(preds_origin - y_origin)
            total_error += abs_error.sum().item()
            total_count += x_batch.size(0)

    avg_error = total_error / total_count
    print(f"[Test] 平均絕對誤差 (磅): {avg_error:.3f}")

    # 可以檢查是否 < 15磅
    if avg_error < 15:
        print("測試結果達到目標範圍內！")
    else:
        print("測試結果仍有改善空間。")


if __name__ == "__main__":
    main()
