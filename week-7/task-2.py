import csv
import math
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============== 1. 資料讀取與前處理 ===============

def load_titanic_data(csv_file):
    """
    讀取 Titanic CSV，取 [Pclass, Sex, Age, Fare, Survived] 作為範例。
    - 若 Age/Fare 缺失值則以平均值填補
    """
    data = []
    ages = []
    fares = []

    # 第一次掃描：收集 age, fare 以計算平均
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳過標題
        for row in reader:
            if len(row) < 10:
                continue
            age_str = row[5].strip()
            fare_str = row[9].strip()
            if age_str != "":
                ages.append(float(age_str))
            if fare_str != "":
                fares.append(float(fare_str))

    avg_age = sum(ages) / len(ages) if len(ages) > 0 else 30.0
    avg_fare = sum(fares) / len(fares) if len(fares) > 0 else 14.0

    # 第二次掃描：完整解析與填補缺失值
    data.clear()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 10:
                continue
            try:
                survived = int(row[1])    # Survived
                pclass   = int(row[2])    # Pclass
                sex_str  = row[4].strip().lower()  # Sex
                age_str  = row[5].strip() # Age
                fare_str = row[9].strip() # Fare

                # Sex 轉成 0/1
                if sex_str == 'male':
                    sex_val = 1.0
                else:
                    sex_val = 0.0

                # Age 填補
                if age_str == "":
                    age_val = avg_age
                else:
                    age_val = float(age_str)

                # Fare 填補
                if fare_str == "":
                    fare_val = avg_fare
                else:
                    fare_val = float(fare_str)

                data.append((pclass, sex_val, age_val, fare_val, survived))
            except:
                continue

    return data

def standardize_column(data_list, col_idx):
    """
    對 data_list 中的 col_idx 進行 z-score 標準化，就地修改 data_list，
    並回傳該欄位的 (mean, std) 以便測試集使用。
    """
    vals = [d[col_idx] for d in data_list]
    mean_val = sum(vals) / len(vals)
    var_val = sum((v - mean_val)**2 for v in vals) / len(vals)
    std_val = math.sqrt(var_val) if var_val > 1e-8 else 1.0

    for i in range(len(data_list)):
        row = list(data_list[i])
        row[col_idx] = (row[col_idx] - mean_val) / std_val
        data_list[i] = tuple(row)

    return mean_val, std_val

def apply_standardize_column(data_list, col_idx, mean_val, std_val):
    """
    測試集要用跟訓練集一樣的 mean/std 來做標準化。
    """
    for i in range(len(data_list)):
        row = list(data_list[i])
        row[col_idx] = (row[col_idx] - mean_val) / std_val
        data_list[i] = tuple(row)


# =============== 2. 自訂 Dataset ===============
class TitanicDataset(Dataset):
    """
    將 (pclass, sex, age, fare, survived) -> (X, y)：
    X = [pclass, sex, age, fare]， y = [survived]
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pclass, sex_val, age_val, fare_val, survived = self.data_list[idx]
        x = torch.tensor([pclass, sex_val, age_val, fare_val], dtype=torch.float32)
        y = torch.tensor([survived], dtype=torch.float32)  # shape: (1,)
        return x, y


# =============== 3. PyTorch 模型定義 ===============
class MyTitanicModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim1=16, hidden_dim2=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()  # 二元分類 [0,1] 輸出
        )

    def forward(self, x):
        return self.net(x)


# =============== 4. 訓練/評估函式 ===============
def train_one_epoch(model, dataloader, loss_fn, optimizer, device='cpu'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # forward
        pred = model(X)       # shape: (batch_size, 1)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

        # 計算訓練正確率
        preds = (pred > 0.5).float()
        correct += (preds == y).sum().item()
        total += X.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def evaluate(model, dataloader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * X.size(0)

            preds = (pred > 0.5).float()
            correct += (preds == y).sum().item()
            total += X.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


# =============== 5. 主程式入口點 ===============
if __name__ == "__main__":
    print("=== [PyTorch] Titanic Survival Prediction with Train/Test Split ===")

    # 固定隨機種子（非必須，但有助於結果可重現）
    random.seed(42)
    torch.manual_seed(42)

    # 1. 讀取 Titanic 資料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "titanic.csv")
    all_data = load_titanic_data(csv_path)
    print(f"Total data size: {len(all_data)}")

    # 2. 洗牌 + 切分 (8:2) 當作 Train/Test
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # 3. 對 train_data 做標準化 (pclass=0, age=2, fare=3)，sex=1不動，survived=4不動
    mean_pclass, std_pclass = standardize_column(train_data, 0)
    mean_age, std_age       = standardize_column(train_data, 2)
    mean_fare, std_fare     = standardize_column(train_data, 3)

    # 對 test_data 套用與 train 相同的標準化參數
    apply_standardize_column(test_data, 0, mean_pclass, std_pclass)
    apply_standardize_column(test_data, 2, mean_age, std_age)
    apply_standardize_column(test_data, 3, mean_fare, std_fare)

    # 4. 建立 Dataset & DataLoader
    train_dataset = TitanicDataset(train_data)
    test_dataset  = TitanicDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 5. 建立模型、損失函數、優化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyTitanicModel(input_dim=4, hidden_dim1=16, hidden_dim2=8).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 6. 開始訓練
    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f}")

    print("=== Training Complete ===")
    print(f"Final Test Accuracy: {test_acc:.3f}")
