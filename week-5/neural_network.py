import math

# =========== Activation Functions ===========
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
    """ 傳入的 sig_val = sigmoid_scalar(z) """
    return sig_val * (1.0 - sig_val)

def sigmoid_vector(z_list):
    """ 針對向量(多個輸出)做 Sigmoid """
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
    return total / len(outputs)

def mse_derivative(output, expect):
    """
    MSE 對單一輸出的導數: 2*(O - E)
    """
    return 2.0 * (output - expect)

def binary_cross_entropy(outputs, expects):
    """
    二元交叉熵 (也用於 Multi-Label):
    BCE = - Σ [ E_i * log(O_i) + (1 - E_i)*log(1 - O_i) ]
    outputs, expects: list of float, 例如 sigmoid 之後的多輸出
    """
    assert len(outputs) == len(expects), "BCE: 輸出/期望維度不符"
    epsilon = 1e-15
    total = 0.0
    for i in range(len(outputs)):
        o_i = min(max(outputs[i], epsilon), 1 - epsilon)  # 避免 log(0)
        e_i = expects[i]
        total += - ( e_i*math.log(o_i) + (1.0 - e_i)*math.log(1.0 - o_i) )
    return total

def bce_derivative(output, expect):
    """
    BCE 對單一輸出的導數:
       d(Loss)/d(O) = -( E/O - (1-E)/(1-O) )
                    = (O - E) / [ O(1 - O) ]
    """
    epsilon = 1e-15
    o_i = min(max(output, epsilon), 1.0 - epsilon)
    e_i = expect
    return -( e_i / o_i - (1.0 - e_i)/(1.0 - o_i) )

# =========== Network Class ===========
class Network:
    def __init__(self,
                 input_size=2,
                 hidden_size=2,
                 output_size=1,
                 hidden_activation=relu,
                 output_activation='linear',
                 w_h=None,
                 w_o=None):
        """
        建立一個簡單的網路：
        - 預設: 2 輸入、2 隱藏
        - output_size 可變
        - hidden_activation / output_activation 可變
        - w_h, w_o 為外部可傳入的權重 (測試或指定用)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation_type = output_activation

        # 如果外部沒傳，就用預設值
        if w_h is None:
            self.weights_h = [
                [0.5,  0.2,  0.3],   # Neuron1
                [0.6, -0.6,  0.25],  # Neuron2
            ]
        else:
            self.weights_h = w_h
            
        if w_o is None:
            self.weights_o = [
                [0.8,  -0.5,  0.6],   # Output neuron1
                [0.4,   0.5, -0.25],  # Output neuron2 (如需多輸出)
            ]
        else:
            self.weights_o = w_o
        
        # Forward pass暫存
        self.z_hidden = [0.0] * self.hidden_size
        self.a_hidden = [0.0] * self.hidden_size
        self.z_output = [0.0] * self.output_size
        self.a_output = [0.0] * self.output_size

        # 梯度暫存
        self.grad_w_h = [[0.0]*3 for _ in range(self.hidden_size)]
        self.grad_w_o = [[0.0]*3 for _ in range(self.output_size)]

    def forward(self, x1, x2):
        # ===== Hidden Layer =====
        for i in range(self.hidden_size):
            w1, w2, b = self.weights_h[i]
            z = w1*x1 + w2*x2 + b
            self.z_hidden[i] = z
            if self.hidden_activation == relu:
                self.a_hidden[i] = relu(z)
            else:
                self.a_hidden[i] = z  # linear

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
        """
        訓練用的 forward, 會順便記錄 x1, x2
        """
        self.x1 = x1
        self.x2 = x2
        return self.forward(x1, x2)

    def backward(self, expects, loss_type='mse'):
        """
        只示範 MSE / BCE
        如果要支援更多，可以在這裡擴充
        """
        # 1. d(Loss)/d(Output)
        dL_dOut = [0.0]*self.output_size
        if loss_type == 'mse':
            # MSE: 2*(O - E)
            for i in range(self.output_size):
                dL_dOut[i] = mse_derivative(self.a_output[i], expects[i])
        elif loss_type == 'bce':
            # BCE: (O - E) / [ O(1 - O) ]
            for i in range(self.output_size):
                dL_dOut[i] = bce_derivative(self.a_output[i], expects[i])
        else:
            raise NotImplementedError("只示範 'mse' 和 'bce'")

        # 2. 乘上輸出層 activation 的導數 => d(Loss)/d(z_output)
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
                dL_dZ_hidden[j] = dL_dA_hidden[j]  # linear derivative=1

        # 6. 隱藏層權重梯度
        x1, x2 = self.x1, self.x2
        for j in range(self.hidden_size):
            self.grad_w_h[j][0] = dL_dZ_hidden[j] * x1
            self.grad_w_h[j][1] = dL_dZ_hidden[j] * x2
            self.grad_w_h[j][2] = dL_dZ_hidden[j] * 1.0

    def zero_grad(self, learning_rate=0.01):
        """
        根據 grad_w_h, grad_w_o 做梯度下降，然後將梯度歸零
        """
        # 更新隱藏層
        for j in range(self.hidden_size):
            for k in range(3):
                self.weights_h[j][k] -= learning_rate * self.grad_w_h[j][k]
                self.grad_w_h[j][k] = 0.0

        # 更新輸出層
        for i in range(self.output_size):
            for k in range(3):
                self.weights_o[i][k] -= learning_rate * self.grad_w_o[i][k]
                self.grad_w_o[i][k] = 0.0

if __name__ == "__main__":
    # ==================================================
    # ==============  Task1: Regression  ===============
    # ==================================================
    print("=== [Task1] Neural Network for Regression Tasks ===")

    # Task1-1:
    # 這裡手動指定 w_h, w_o 與您題目圖片相同的初始值
    # （2個隱藏神經元 + 2個輸出神經元）
    custom_w_h = [
        [0.6, 0.2, 0.3],  # Hidden Neuron1
        [0.3, 0.5, 0.25], # Hidden Neuron2
    ]
    custom_w_o = [
        [0.8,  -0.5,  0.6],   # Output neuron O1
        [0.4,  -0.3,  0.75],  # Output neuron O2
    ]

    # 建立網路: hidden_size=2, output_size=2, output_activation='linear'
    # 並套用圖中權重
    nn_reg = Network(
        input_size=2,
        hidden_size=2,
        output_size=2,
        hidden_activation=relu,
        output_activation='linear',
        w_h=custom_w_h,
        w_o=custom_w_o
    )

    # 指定輸入、期望
    x1, x2 = (1.5, 0.5)
    expects = [0.8, 1.0]

    # Forward → Loss → Backward → Update (lr=0.01)
    outputs = nn_reg.forward_train(x1, x2)
    loss = mse_loss(outputs, expects)
    print(f"[Task1-1] Initial Outputs={outputs}, MSE Loss={loss:.4f}")

    nn_reg.backward(expects, loss_type='mse')
    nn_reg.zero_grad(learning_rate=0.01)

    # 印出更新後的權重
    print("[Task1-1] Updated weights_h =", nn_reg.weights_h)
    print("[Task1-1] Updated weights_o =", nn_reg.weights_o)

    # Task1-2:
    # 在同一筆資料 (1.5, 0.5) => (0.8, 1.0) 上，重複 1000 次
    epochs = 1000
    for epoch in range(epochs):
        outputs = nn_reg.forward_train(x1, x2)
        loss = mse_loss(outputs, expects)
        nn_reg.backward(expects, 'mse')
        nn_reg.zero_grad(learning_rate=0.01)

    print(f"\n[Task1-2] After {epochs} epochs, final outputs={outputs}, final loss={loss:.8f}")
    print("Loss應該非常接近0，表示回歸問題收斂良好。")

    # ==================================================
    # ============  Task2: Binary Classification  ======
    # ==================================================
    print("\n=== [Task2] Neural Network for Binary Classification Tasks ===")

    # Task2-1:
    nn_bin = Network(output_size=1, output_activation='sigmoid',
                     w_o=[[0.8, 0.4, -0.5]])  # 題目示例

    x1, x2 = (0.75, 1.25)
    expects = [1]

    outputs = nn_bin.forward_train(x1, x2)
    loss = binary_cross_entropy(outputs, expects)
    print(f"[Task2-1] Initial Output={outputs}, BCE Loss={loss:.4f}")

    nn_bin.backward(expects, loss_type='bce')
    nn_bin.zero_grad(learning_rate=0.1)

    print("[Task2-1] Updated weights_h =", nn_bin.weights_h)
    print("[Task2-1] Updated weights_o =", nn_bin.weights_o)

    # Task2-2:
    epochs = 1000
    for epoch in range(epochs):
        outputs = nn_bin.forward_train(x1, x2)
        loss = binary_cross_entropy(outputs, expects)
        nn_bin.backward(expects, 'bce')
        nn_bin.zero_grad(learning_rate=0.1)

    print(f"\n[Task2-2] After {epochs} epochs, final output={outputs}, final BCE loss={loss:.8f}")
    print("Loss應該非常接近0，表示二元分類問題收斂良好。")
