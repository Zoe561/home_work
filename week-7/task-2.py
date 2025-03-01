import csv
import math
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============== 1. 資料讀取與前處理 (與原程式類似) ===============

def load_titanic_data(csv_file):
    """
    讀取 Titanic CSV，取 [Pclass, Sex, Age, Fare, Survived] 五欄作為範例。
    - 若 Age/Fare 缺失值則以平均值填補
    """
    data = []
    ages = []
    fares = []

    # 先掃描一次 CSV，把 Age / Fare 收集起來，以便後面計算平均
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                # Survived = row[1], Pclass = row[2], Sex = row[4], Age = row[5], Fare=row[9]
                age_str = row[5].strip()
                fare_str = row[9].strip()
                if age_str != "":
                    ages.append(float(age_str))
                if fare_str != "":
                    fares.append(float(fare_str))
            except:
                pass

    avg_age = sum(ages) / len(ages) if len(ages) > 0 else 30.0
    avg_fare = sum(fares) / len(fares) if len(fares) > 0 else 14.0

    # 再重新讀一次，完整解析
    data.clear()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                survived = int(row[1])
                pclass = int(row[2])
                sex_str = row[4].strip().lower()
                age_str = row[5].strip()
                fare_str = row[9].strip()

                if sex_str == 'male':
                    sex_val = 1.0
                else:
                    sex_val = 0.0

                if age_str == "":
                    age_val = avg_age
                else:
                    age_val = float(age_str)

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
    對 data_list 中指定 col_idx 進行 z-score 標準化，
    會就地修改 data_list，並回傳 (mean, std)，以便在測試集上使用同樣的 mean/std。
    """
    vals = [d[col_idx] for d in data_list]
    mean_val = sum(vals)/len(vals)
    var_val = sum((v - mean_val)**2 for v in vals)/len(vals)
    std_val = math.sqrt(var_val) if var_val > 0 else 1.0

    for i in range(len(data_list)):
        row = list(data_list[i])
        row[col_idx] = (row[col_idx] - mean_val)/std_val
        data_list[i] = tuple(row)

    return mean_val, std_val

def apply_standardize_column(data_list, col_idx, mean_val, std_val):
    """
    把指定 col_idx 用給定的 mean_val, std_val 標準化。
    """
    for i in range(len(data_list)):
        row = list(data_list[i])
        row[col_idx] = (row[col_idx] - mean_val)/std_val
        data_list[i] = tuple(row)


# =============== 2. 自訂 Dataset ===============
class TitanicDataset(Dataset):
    """
    將 (pclass, sex, age, fare, survived) 對應成 (X, y)：
    X = [pclass, sex, age, fare]，y = survived(0 or 1)
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list  # list of tuples

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pclass, sex_val, age_val, fare_val, survived = self.data_list[idx]

        # X: 4 維特徵
        X = torch.tensor([pclass, sex_val, age_val, fare_val], dtype=torch.float32)
        # y: label
        y = torch.tensor([survived], dtype=torch.float32)
        return X, y


# =============== 3. 定義 PyTorch 模型 ===============
class MyTitanicModel(nn.Module):
    def __init__(self):
        super(MyTitanicModel, self).__init__()
        # 輸入 4 維 -> 隱藏層大小可自行調整
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 二元分類輸出 [0,1]
        )

    def forward(self, x):
        return self.net(x)


# =============== 4. 訓練與測試流程 ===============
def train_one_epoch(model, dataloader, loss_fn, optimizer, device='cpu'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

        # 計算訓練正確率
        # 預測值 pred.shape = (batch_size, 1)
        # 若 pred > 0.5 則視為生存(1)，否則(0)
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
    print("=== [PyTorch] Titanic Survival Prediction ===")

    # 固定隨機種子（非必須，但有助於結果可重現）
    random.seed(42)
    torch.manual_seed(42)

    # 1. 讀資料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    titanic_path = os.path.join(script_dir, "titanic.csv")
    dataset_all = load_titanic_data(titanic_path)

    # 2. 洗牌 + 切分訓練 / 測試集
    random.shuffle(dataset_all)
    split_idx = int(len(dataset_all) * 0.8)
    train_data = dataset_all[:split_idx]
    test_data = dataset_all[split_idx:]

    # 3. 對 Pclass(0), Age(2), Fare(3) 做標準化；Sex(1) 為 0/1 不做
    mean_pclass, std_pclass = standardize_column(train_data, 0)
    mean_age, std_age = standardize_column(train_data, 2)
    mean_fare, std_fare = standardize_column(train_data, 3)

    # 測試集套用相同 mean/std
    apply_standardize_column(test_data, 0, mean_pclass, std_pclass)
    apply_standardize_column(test_data, 2, mean_age, std_age)
    apply_standardize_column(test_data, 3, mean_fare, std_fare)

    # 4. 透過自訂 Dataset 與 DataLoader 組織資料
    train_dataset = TitanicDataset(train_data)
    test_dataset = TitanicDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 5. 建立模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyTitanicModel().to(device)

    # 6. 定義 loss function 與 optimizer
    loss_fn = nn.BCELoss()  # 二元分類常用
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 7. 訓練
    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f}")

    # 最後輸出整體測試正確率
    print(f"[Final Test] Accuracy: {test_acc:.3f}")
