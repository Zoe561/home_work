import csv
import math
import random
import os
from network import Network, relu  # 引用 network.py

def load_titanic_data(csv_file):
    """
    簡易讀取 Titanic CSV，取 [Pclass, Sex, Age, Fare, Survived] 五欄作為範例
    - 若有缺失值 (Age等)，示範用平均值或其他方式填補
    """
    data = []
    ages = []
    fares = []

    # 先讀一次，把 Age / Fare 收集起來以便後面計算平均
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 可能包含: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        for row in reader:
            # 假設我們知道各欄的index，根據實際 csv header 來調整
            # Survived = row[1], Pclass = row[2], Sex = row[4], Age = row[5], Fare=row[9]
            try:
                pclass = int(row[2])       # 1,2,3
                sex = row[4].strip().lower()  # male/female
                age_str = row[5].strip()
                fare_str = row[9].strip()
                if age_str != "":
                    ages.append(float(age_str))
                if fare_str != "":
                    fares.append(float(fare_str))
            except:
                # 有些資料格式不符合就跳過
                pass

    avg_age = sum(ages)/len(ages) if len(ages)>0 else 30.0
    avg_fare = sum(fares)/len(fares) if len(fares)>0 else 14.0

    # 重新讀取，完整解析
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
    對 data_list 中指定 col_idx 做 z-score
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

def run_titanic_classification():
    # 1. 讀資料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    titanic_path = os.path.join(script_dir, "titanic.csv")
    dataset = load_titanic_data(titanic_path)  # 檔名依實際情況

    # 2. 洗牌 + 分割訓練/測試
    random.shuffle(dataset)
    split_idx = int(len(dataset)*0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # 3. 對 Pclass, Age, Fare 做標準化 (Sex 通常是 0/1 不用)
    #    (pclass=0, sex=1, age=2, fare=3, label=4)
    standardize_column(train_data, 0)  # Pclass
    standardize_column(train_data, 2)  # Age
    standardize_column(train_data, 3)  # Fare

    # 測試集用同樣 mean/std (這裡簡化處理，直接重新讀 mean/std)
    # 若要最精準，可將 mean/std 存起來再套用
    mean_pclass, std_pclass = standardize_column(test_data, 0)
    mean_age, std_age = standardize_column(test_data, 2)
    mean_fare, std_fare = standardize_column(test_data, 3)

    # 4. 建立網路 (輸入有4維: pclass, sex, age, fare；輸出1維: survived或否)
    nn = Network(input_size=2,
                 hidden_size=2,  # 可自行調參
                 output_size=1,
                 hidden_activation=relu,
                 output_activation='sigmoid')

    # 5. 訓練
    epochs = 10
    lr = 0.01
    for ep in range(epochs):
        for row in train_data:
            pclass, sex_val, age_val, fare_val, survived = row
            outputs = nn.forward_train(pclass, sex_val)  # 但注意: 我們只有 forward 2 參數?
            # ----
            # !! 這裡要特別注意: 目前 Network 類別是兩個輸入 x1, x2
            # 我們有4個輸入 => 需自行改 Network forward() 使其可處理多維輸入
            # (題目給的簡易版只示範2 input)
            # ----
            # 暫時示範: 只放 pclass, sex_val(2維) => 準確度不會太高
            # 如果要4維, 要擴增 forward() 與權重
            # ----
            nn.backward([float(survived)], loss_type='bce')
            nn.zero_grad(learning_rate=lr)

        # 計算訓練集正確率
        correct_count = 0
        for row in train_data:
            pclass, sex_val, age_val, fare_val, survived = row
            out = nn.forward(pclass, sex_val)[0]
            pred = 1 if out > 0.5 else 0
            if pred == survived:
                correct_count += 1
        train_acc = correct_count / len(train_data)
        print(f"Epoch {ep+1}/{epochs} | Train Accuracy: {train_acc:.3f}")

    # 6. 測試集正確率
    correct_count = 0
    for row in test_data:
        pclass, sex_val, age_val, fare_val, survived = row
        out = nn.forward(pclass, sex_val)[0]
        pred = 1 if out > 0.5 else 0
        if pred == survived:
            correct_count += 1
    test_acc = correct_count / len(test_data)
    print(f"[Test] Accuracy: {test_acc:.3f}")

# ------------------------------------------------------
# 主程式入口
if __name__ == "__main__":
    print("=== [Task2] Classification: Titanic Survival Prediction ===")
    run_titanic_classification()