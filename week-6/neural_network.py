import csv
import math
import random
import os


# =========== Activation / Loss / Network ===========

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def linear(x):
    return x

def linear_derivative(_):
    return 1.0

def sigmoid_scalar(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(sig_val):
    return sig_val * (1.0 - sig_val)

def bce_derivative(output, expect):
    epsilon = 1e-15
    o_i = min(max(output, epsilon), 1.0 - epsilon)
    e_i = expect
    return -( e_i / o_i - (1.0 - e_i)/(1.0 - o_i) )

def mse_derivative(output, expect):
    return 2.0 * (output - expect)

class Network:
    def __init__(self,
                 input_size=2,
                 hidden_size=2,
                 output_size=1,
                 hidden_activation=relu,
                 output_activation='linear',
                 w_h=None,
                 w_o=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation_type = output_activation

        # 如果外部沒傳，就用預設值 (可自行初始化)
        if w_h is None:
            # hidden_size x (input_size + 1)
            # 範例預設 2 個隱藏神經元
            self.weights_h = [
                [0.5,  0.2,  0.3],
                [0.6, -0.6,  0.25],
            ]
        else:
            self.weights_h = w_h
            
        if w_o is None:
            # output_size x (hidden_size + 1)
            # 範例預設 1~2 輸出神經元
            self.weights_o = [
                [0.8, -0.5, 0.6],  # 如果只要 1 輸出就用第一行即可
            ]
        else:
            self.weights_o = w_o
        
        # Forward pass暫存
        self.z_hidden = [0.0] * self.hidden_size
        self.a_hidden = [0.0] * self.hidden_size
        self.z_output = [0.0] * self.output_size
        self.a_output = [0.0] * self.output_size

        # 梯度暫存
        self.grad_w_h = [[0.0]*(self.input_size+1) for _ in range(self.hidden_size)]
        self.grad_w_o = [[0.0]*(self.hidden_size+1) for _ in range(self.output_size)]

    def forward(self, x1, x2):
        # ===== Hidden Layer =====
        for i in range(self.hidden_size):
            w1, w2, b = self.weights_h[i]
            z = w1*x1 + w2*x2 + b
            self.z_hidden[i] = z
            # activation
            if self.hidden_activation == relu:
                self.a_hidden[i] = relu(z)
            else:
                self.a_hidden[i] = z  # linear or other

        # ===== Output Layer =====
        raw_outputs = []
        for i in range(self.output_size):
            w_h1, w_h2, b = self.weights_o[i]
            z = w_h1*self.a_hidden[0] + w_h2*self.a_hidden[1] + b
            self.z_output[i] = z
            raw_outputs.append(z)

        # ===== Output Activation =====
        final_outputs = []
        if self.output_activation_type == 'linear':
            final_outputs = [linear(z) for z in raw_outputs]
        elif self.output_activation_type == 'sigmoid':
            final_outputs = [sigmoid_scalar(z) for z in raw_outputs]
        else:
            raise ValueError("Unsupported output_activation")

        for i in range(self.output_size):
            self.a_output[i] = final_outputs[i]

        return final_outputs

    def forward_train(self, x1, x2):
        # 訓練用 forward，順便記錄 x1, x2
        self.x1 = x1
        self.x2 = x2
        return self.forward(x1, x2)

    def backward(self, expects, loss_type='mse'):
        # 1. d(Loss)/d(Output)
        dL_dOut = [0.0]*self.output_size
        if loss_type == 'mse':
            for i in range(self.output_size):
                dL_dOut[i] = mse_derivative(self.a_output[i], expects[i])
        elif loss_type == 'bce':
            for i in range(self.output_size):
                dL_dOut[i] = bce_derivative(self.a_output[i], expects[i])
        else:
            raise NotImplementedError("只示範 'mse' 和 'bce'")

        # 2. 輸出層 activation 導數 => d(Loss)/d(z_output)
        dL_dZ_out = [0.0]*self.output_size
        for i in range(self.output_size):
            if self.output_activation_type == 'linear':
                dL_dZ_out[i] = dL_dOut[i] * linear_derivative(self.z_output[i])
            elif self.output_activation_type == 'sigmoid':
                sig_val = self.a_output[i]
                dL_dZ_out[i] = dL_dOut[i] * sigmoid_derivative(sig_val)

        # 3. Output layer 的權重梯度
        for i in range(self.output_size):
            self.grad_w_o[i][0] = dL_dZ_out[i] * self.a_hidden[0]
            self.grad_w_o[i][1] = dL_dZ_out[i] * self.a_hidden[1]
            self.grad_w_o[i][2] = dL_dZ_out[i] * 1.0

        # 4. 反傳到隱藏層
        dL_dA_hidden = [0.0]*self.hidden_size
        for j in range(self.hidden_size):
            s = 0.0
            for i in range(self.output_size):
                w_oij = self.weights_o[i][j]  # j=0 or 1
                s += dL_dZ_out[i] * w_oij
            dL_dA_hidden[j] = s

        # 5. 隱藏層 activation 導數 => d(Loss)/d(z_hidden)
        dL_dZ_hidden = [0.0]*self.hidden_size
        for j in range(self.hidden_size):
            if self.hidden_activation == relu:
                d_act = relu_derivative(self.z_hidden[j])
                dL_dZ_hidden[j] = dL_dA_hidden[j] * d_act
            else:
                dL_dZ_hidden[j] = dL_dA_hidden[j]

        # 6. 隱藏層權重梯度
        x1, x2 = self.x1, self.x2
        for j in range(self.hidden_size):
            self.grad_w_h[j][0] = dL_dZ_hidden[j] * x1
            self.grad_w_h[j][1] = dL_dZ_hidden[j] * x2
            self.grad_w_h[j][2] = dL_dZ_hidden[j] * 1.0

    def zero_grad(self, learning_rate=0.01):
        # 根據 grad_w_h, grad_w_o 做梯度下降，然後將梯度清零
        for j in range(self.hidden_size):
            for k in range(self.input_size+1):
                self.weights_h[j][k] -= learning_rate * self.grad_w_h[j][k]
                self.grad_w_h[j][k] = 0.0
        for i in range(self.output_size):
            for k in range(self.hidden_size+1):
                self.weights_o[i][k] -= learning_rate * self.grad_w_o[i][k]
                self.grad_w_o[i][k] = 0.0


# ========== 作業 1: 預測體重(Regression) ==========

def load_weight_height_data(csv_file):
    """
    讀取 weight-height 資料，並回傳 list of (gender_value, height, weight)
    假設每一行格式: Gender,Height,Weight
    """
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳過標頭
        for row in reader:
            if len(row) < 3:
                continue
            gender_str = row[0].strip()
            height = float(row[1])
            weight = float(row[2])
            # 性別編碼: Male -> 1, Female -> 0
            gender_val = 1.0 if gender_str.lower().startswith('m') else 0.0
            data.append((gender_val, height, weight))
    return data

def standardize(data_list, idx):
    """
    對 data_list (list of tuples) 中第 idx 欄做 z-score 標準化
    回傳 mean, std 以便後續反向轉回原尺度
    """
    vals = [d[idx] for d in data_list]
    mean_val = sum(vals) / len(vals)
    var_val = sum((v - mean_val)**2 for v in vals) / len(vals)
    std_val = math.sqrt(var_val)

    if std_val < 1e-12:
        std_val = 1.0  # 避免分母為 0

    # 就地修改 data_list
    for i in range(len(data_list)):
        tmp = list(data_list[i])
        tmp[idx] = (tmp[idx] - mean_val) / std_val
        data_list[i] = tuple(tmp)

    return mean_val, std_val

def run_regression_task():
    # 1. 載入資料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    height_weight_path = os.path.join(script_dir, "gender-height-weight.csv")

    dataset = load_weight_height_data(height_weight_path)

    # 2. 資料打亂 (隨機洗牌)
    random.shuffle(dataset)

    # 3. 分割 訓練/測試
    split_idx = int(len(dataset)*0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # 4. 將 height, weight 做標準化 (gender 可視情況決定是否也要標準化)
    mean_h, std_h = standardize(train_data, 1)
    mean_w, std_w = standardize(train_data, 2)

    # 測試資料同樣以「訓練集的 mean/std」做標準化
    for i in range(len(test_data)):
        g, h, w = test_data[i]
        h = (h - mean_h)/std_h
        w = (w - mean_w)/std_w
        test_data[i] = (g, h, w)

    # 5. 準備網路
    nn = Network(input_size=2,  # [gender, height]
                 hidden_size=2, # 可自行調整
                 output_size=1,
                 hidden_activation=relu,      # 使用 ReLU
                 output_activation='linear') # 輸出做回歸

    # 6. 進行訓練
    epochs = 5  # 可加大，例如 50, 100, etc.
    lr = 0.01
    for ep in range(epochs):
        for (g, h, w) in train_data:
            outputs = nn.forward_train(g, h)
            # expects = [w]
            nn.backward([w], loss_type='mse')
            nn.zero_grad(learning_rate=lr)

        # 每個 epoch 簡單看一下在訓練集的平均誤差
        train_loss_sum = 0.0
        for (g, h, w) in train_data:
            pred = nn.forward(g, h)[0]
            diff = (w - pred)**2  # MSE component
            train_loss_sum += diff
        train_mse = train_loss_sum / len(train_data)
        print(f"Epoch {ep+1}/{epochs}, Train MSE: {train_mse:.4f}")

    # 7. 在測試集評估 (計算平均誤差 及 反向轉回原單位)
    err_sum = 0.0
    for (g, h, w) in test_data:
        pred = nn.forward(g, h)[0]  # 預測 (標準化後)
        # 反標準化: w' = pred*std_w + mean_w
        pred_pound = pred*std_w + mean_w
        real_pound = w*std_w + mean_w
        err_sum += abs(pred_pound - real_pound)

    avg_err = err_sum / len(test_data)
    print(f"[Test] Average Error in Pounds: {avg_err:.3f}")

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
    print("=== [Task1] Regression: Predict Weight by Gender & Height ===")
    run_regression_task()

    print("=== [Task2] Classification: Titanic Survival Prediction ===")
    run_titanic_classification()
