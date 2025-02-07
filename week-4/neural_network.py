class Network:
    def __init__(self):
        """
        初始化網路權重：
          隱藏層(2 neurons):
            H1_in = 0.5*x1 + 0.2*x2 + 0.3*bias
            H2_in = 0.6*x1 + (-0.6)*x2 + 0.25*bias
          輸出層(2 neurons):
            O1 = 0.8*H1 + (-0.5)*H2 + 0.6*bias
            O2 = (-0.5)*H1 + 0.5*H2 + (-0.25)*bias
        """
        # 隱藏層權重: (w1, w2, b)
        self.w_h = [
            [0.5,  0.2,  0.3],   # H1 的 (x1, x2, bias)
            [0.6, -0.6,  0.25]   # H2 的 (x1, x2, bias)
        ]
        # 輸出層權重: (wH1, wH2, b)
        self.w_o = [
            [0.8,  -0.5,  0.6],   # O1
            [-0.5,  0.5, -0.25]   # O2
        ]

    def relu(self, x):
        return x if x > 0 else 0

    def forward(self, x1, x2):
        """
        前向傳播流程：
          1) 計算隱藏層的加權和 + ReLU
          2) 計算輸出層的加權和 (Linear)
          傳回 [O1, O2]
        """
        # ------
        # 隱藏層
        # ------
        hidden_outputs = []
        for (w1, w2, b) in self.w_h:
            z = w1 * x1 + w2 * x2 + b
            h = self.relu(z)
            hidden_outputs.append(h)  # [H1, H2]

        # ------
        # 輸出層 (線性)
        # ------
        outputs = []
        for (wH1, wH2, b) in self.w_o:
            z = wH1 * hidden_outputs[0] + wH2 * hidden_outputs[1] + b
            outputs.append(z)  # [O1, O2]

        return outputs

def mean_squared_error(outputs, expects):
    """
    MSE = (1/n) * Σ (E_i - O_i)^2
    outputs, expects 皆為 [o1, o2], [e1, e2] 形式
    """
    assert len(outputs) == len(expects), "outputs 與 expects 維度必須相同"
    n = len(outputs)
    mse = 0.0
    for i in range(n):
        mse += (expects[i] - outputs[i])**2
    return mse / n


if __name__ == "__main__":
    # 建立 Network
    nn = Network()

    # 第1筆資料
    outputs = nn.forward(1.5, 0.5)
    expects = [0.8, 1.0]
    print("Total Loss", mean_squared_error(outputs, expects))

    # 第2筆資料
    outputs = nn.forward(0, 1)
    expects = [0.5, 0.5]
    print("Total Loss", mean_squared_error(outputs, expects))
