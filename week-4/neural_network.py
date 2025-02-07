import math

# =========== Activation Functions ===========

def relu(x):
    return max(0, x)

def linear(x):
    return x

def sigmoid_scalar(x):
    """針對單一標量做 Sigmoid"""
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_vector(z_list):
    """針對向量(多個輸出)做 Sigmoid"""
    return [sigmoid_scalar(v) for v in z_list]

def softmax(z_list):
    """
    針對向量做 softmax, 避免 overflow:
      softmax(x_i) = exp(x_i - max_x) / Σ exp(x_k - max_x)
    """
    max_z = max(z_list)
    exps = [math.exp(v - max_z) for v in z_list]
    sum_exps = sum(exps)
    return [v / sum_exps for v in exps]


# =========== Loss Functions ===========

def mse_loss(outputs, expects):
    """
    MSE = (1/n) * Σ (E_i - O_i)^2
    outputs, expects: list of float (回歸情況、或多維回歸)
    """
    assert len(outputs) == len(expects), "MSE: 輸出/期望維度不符"
    total = 0.0
    for i in range(len(outputs)):
        total += (expects[i] - outputs[i])**2
    return total

def binary_cross_entropy(outputs, expects):
    """
    二元交叉熵 (也用於 Multi-Label):
    BCE = - Σ [ E_i * log(O_i) + (1 - E_i)*log(1 - O_i) ]
    outputs, expects: list of float, e.g. Sigmoid 之後的多輸出
    """
    assert len(outputs) == len(expects), "BCE: 輸出/期望維度不符"
    epsilon = 1e-15
    total = 0.0
    for i in range(len(outputs)):
        o_i = min(max(outputs[i], epsilon), 1 - epsilon)  # 避免 log(0)
        e_i = expects[i]
        total += - ( e_i * math.log(o_i) + (1.0 - e_i) * math.log(1.0 - o_i) )
    return total

def categorical_cross_entropy(outputs, expects):
    """
    分類交叉熵 (用於 Multi-Class):
    CCE = - Σ [ E_i * log(O_i) ]
    outputs, expects: list of float, 
      其中 outputs 是經 Softmax 後的機率
      expects 是 One-hot，例如 [1, 0, 0]
    """
    assert len(outputs) == len(expects), "CCE: 輸出/期望維度不符"
    epsilon = 1e-15
    total = 0.0
    for i in range(len(outputs)):
        o_i = max(outputs[i], epsilon)  # 避免 log(0)
        e_i = expects[i]
        total += - e_i * math.log(o_i)
    return total


# =========== Network Class ===========

class Network:
    def __init__(self,
                 input_size=2,
                 hidden_size=2,
                 output_size=1,
                 hidden_activation=relu,
                 output_activation='linear',
                 w_h=None, w_o=None
                 ):
        """
        建立一個簡單的網路：
        - input -> hidden(一層) -> output
        - 預設: 2 輸入、2 隱藏、1 輸出
        - 可以依情境，改動 output_size & output_activation
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_activation = hidden_activation
        self.output_activation_type = output_activation

       # 如果外部沒傳，就用預設值
        if w_h is None:
            # 預設隱藏層權重
            self.weights_h = [
                [0.5,  0.2,  0.3],   # Neuron 1
                [0.6, -0.6,  0.25],  # Neuron 2
            ]
        else:
            self.weights_h = w_h
            
        if w_o is None:
            # 預設輸出層權重
            self.weights_o = [
                [0.8,  -0.5,   0.6],   # Output neuron 1
                [0.4,  0.5,  -0.25],  # Output neuron 2
                [0.4,  -0.4,   0.75],  # Output neuron 3 (可供多輸出場景)
            ]
        else:
            self.weights_o = w_o
            
    def forward(self, x1, x2):
        """前向傳播: 計算 hidden -> output -> 最終 activation"""
        # ========== Hidden Layer ==========
        # hidden_size 預設 2, 所以取 self.weights_h 的前 hidden_size 筆
        hidden_outputs = []
        for i in range(self.hidden_size):
            w1, w2, b = self.weights_h[i]
            z = w1*x1 + w2*x2 + b
            h = self.hidden_activation(z)
            hidden_outputs.append(h)

        # ========== Output Layer (linear) ==========
        # 先算線性加權和 z，再根據 output_activation_type 做最終處理
        raw_outputs = []
        for i in range(self.output_size):
            w_h1, w_h2, b = self.weights_o[i]
            # hidden_outputs[0], hidden_outputs[1], + bias
            z = w_h1 * hidden_outputs[0] + w_h2 * hidden_outputs[1] + b
            raw_outputs.append(z)

        # ========== 最終 Activation ==========
        if self.output_activation_type == 'linear':
            final_outputs = [linear(z) for z in raw_outputs]
        elif self.output_activation_type == 'sigmoid':
            # 二元分類(單一輸出)或多標籤(多輸出) => 對每個值做 sigmoid
            final_outputs = sigmoid_vector(raw_outputs)
        elif self.output_activation_type == 'softmax':
            # 多分類 => 對整個向量做 softmax
            final_outputs = softmax(raw_outputs)
        else:
            raise ValueError(f"不支援的輸出層 activation: {self.output_activation_type}")

        return final_outputs


if __name__ == "__main__":
    # ============ [範例1] 回歸 (Regression) ============
    print("=== 回歸 (Regression) ===")
    # 建立網路: 輸出層=2, output_activation='linear'
    nn_reg = Network(output_size=2, output_activation='linear')
    # forward
    outputs_reg1 = nn_reg.forward(1.5, 0.5)
    # 假設期望值(單一回歸輸出)
    expects_reg1 = [0.8, 1.0]
    # 計算 MSE
    loss_reg1 = mse_loss(outputs_reg1, expects_reg1)
    # print("Outputs:", outputs_reg, "MSE Loss:", loss_reg)
    print("Total Loss", loss_reg1)

    # forward
    outputs_reg2 = nn_reg.forward(0, 1)
    # 假設期望值(單一回歸輸出)
    expects_reg2 = [0.5, 0.5]
    # 計算 MSE
    loss_reg2 = mse_loss(outputs_reg2, expects_reg2)
    print("Total Loss", loss_reg2)

    # ============ [範例2] 二元分類 (Binary Classification) ============
    print("\n=== 二元分類 (Binary) ===")
    # 建立網路: 輸出層=1, output_activation='sigmoid'
    nn_bin = Network(output_size=1, output_activation='sigmoid', w_o=[[0.8,0.4,-0.5]])
    # forward
    outputs_bin1 = nn_bin.forward(0.75, 1.25)  # e.g. x1=0, x2=1
    expects_bin1 = [1]  # 期望標籤=1
    loss_bin1 = binary_cross_entropy(outputs_bin1, expects_bin1)
    # print("Outputs:", outputs_bin, "BCE Loss:", loss_bin)
    print("Total Loss", loss_bin1)
    
    # forward
    outputs_bin2 = nn_bin.forward(-1, 0.5)  # e.g. x1=0, x2=1
    expects_bin2 = [0]
    loss_bin2 = binary_cross_entropy(outputs_bin2, expects_bin2)
    print("Total Loss", loss_bin2)

    # ============ [範例3] 多分類 (Multi-Class) ============
    print("\n=== 多分類 (Multi-Class) ===")
    # 假設有 3 類 => output_size=3, output_activation='softmax'
    nn_mc = Network(output_size=3, output_activation='softmax')
    # forward
    outputs_mc = nn_mc.forward(1.5, 0.5)
    # 假設期望: 類別為第 0 類 => One-hot=[1, 0, 0]
    expects_mc = [1, 0, 0]
    loss_mc = categorical_cross_entropy(outputs_mc, expects_mc)
    # print("Outputs (softmax):", outputs_mc, "CCE Loss:", loss_mc)
    print("Total Loss", loss_mc)

    # ============ [範例4] 多標籤 (Multi-Label) ============
    print("\n=== 多標籤 (Multi-Label) ===")
    # 假設有 3 個標籤 => output_size=3, output_activation='sigmoid'
    nn_ml = Network(output_size=3, output_activation='sigmoid')
    # forward
    outputs_ml = nn_ml.forward(0, 1)
    # 假設期望標籤: (1, 0, 1)
    expects_ml = [1, 0, 1]
    loss_ml = binary_cross_entropy(outputs_ml, expects_ml)
    # print("Outputs (sigmoid):", outputs_ml, "Multi-Label BCE Loss:", loss_ml)
    print("Total Loss", loss_ml)

