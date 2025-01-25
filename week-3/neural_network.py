class Network:
    def __init__(self, layer_weights, layer_biases):
        """
        layer_weights: list of 2D lists, 每一層的權重矩陣
                       shape = (前一層輸出維度, 本層神經元數)
        layer_biases:  list of 1D lists, 每一層的偏差
                       shape = (本層神經元數,)
        """
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases
        
    def forward(self, inputs):
        # current_outputs 一開始就是輸入層的輸出 (tuple or list)
        current_outputs = inputs
        
        for layer_idx in range(len(self.layer_weights)):
            weights = self.layer_weights[layer_idx]  # shape=(in_dim, out_dim)
            biases = self.layer_biases[layer_idx]    # shape=(out_dim,)
            next_outputs = []
            
            # 對該層「每一個神經元」做加權 + 偏差
            out_dim = len(weights[0])    # 本層神經元數
            in_dim  = len(current_outputs)  # 上一層輸出數 (即本層輸入數)
            
            for neuron_idx in range(out_dim):
                val = 0
                # current_outputs[i] * weights[i][neuron_idx]
                for input_idx in range(in_dim):
                    val += current_outputs[input_idx] * weights[input_idx][neuron_idx]
                val += biases[neuron_idx]
                
                next_outputs.append(val)
            
            # 更新，進入下一層
            current_outputs = next_outputs
            
        return current_outputs


# ---------------------------
# Network 1: 2 輸入 → 2 隱藏 → 1 輸出
# ---------------------------

weights1 = [
    # 第一層(2→2): shape=(2,2)
    [
        [0.5,  0.6],    # x1 -> (hidden1, hidden2)
        [0.2, -0.6]     # x2 -> (hidden1, hidden2)
    ],
    # 第二層(2→1): shape=(2,1)
    [
        [0.8],          # hidden1 -> output
        [0.4]           # hidden2 -> output
    ]
]
biases1 = [
    [0.3, 0.25],  # 第一層 2 個 neuron 的 bias
    [-0.5]        # 第二層 1 個 neuron 的 bias
]

nn1 = Network(weights1, biases1)


# ---------------------------
# Network 2: 2 輸入 → 2 隱藏 → 1 隱藏 → 2 輸出
#   (共 3 層 transform，所以有三組 weights/biases)
# ---------------------------

weights2 = [
    # 第一層 (2→2)
    [
        [0.5,  0.6],   # x1 -> (H1, H2)
        [1.5, -0.8]    # x2 -> (H1, H2)
    ],
    # 第二層 (2→1)
    [
        [0.6],         # H1 -> H3
        [-0.8]         # H2 -> H3
    ],
    # 第三層 (1→2)
    [
        [0.5, -0.4]     # H3 -> (O1, O2)
    ]
]
biases2 = [
    [0.3,  1.25],   # 第一層: (H1, H2)
    [0.3],          # 第二層: (H3)
    [0.2,  0.5]    # 第三層: (O1, O2)
]


nn2 = Network(weights2, biases2)


# ---------------------------------------------------------
# 測試 ：3 輸入 → 3 隱藏 → 2 輸出
# ---------------------------------------------------------

weights3 = [
    # 第 1 層 (3→3)
    [
        [ 1.0,  0.5, -0.2],  # x1 -> (h1, h2, h3)
        [ 0.0, -0.3,  1.0],  # x2 -> (h1, h2, h3)
        [ 2.0,  0.1,  0.9]   # x3 -> (h1, h2, h3)
    ],
    # 第 2 層 (3→2)
    [
        [ 0.2,  1.0],  # h1 -> (out1, out2)
        [ 0.5,  0.7],  # h2 -> (out1, out2)
        [-0.1,  0.3]   # h3 -> (out1, out2)
    ]
]
biases3 = [
    [ 0.1, -0.2,  0.3],   # 第 1 層 3 個 neuron 的偏置
    [ 1.0,  0.5 ]         # 第 2 層 2 個 neuron 的偏置
]

nn3 = Network(weights3, biases3)

# -----------------------------------------------------------
# 測試： 輸入(2) → 隱藏層1(2) → 隱藏層2(2) → 隱藏層3(2) → 輸出(2)
# -----------------------------------------------------------
weights_3HL = [
    # 第 1 層 (2→2)
    [
        [0.1,  0.2],   # x1 -> (h1_1, h1_2)
        [0.3,  0.4]    # x2 -> (h1_1, h1_2)
    ],
    # 第 2 層 (2→2)
    [
        [0.5, -0.2],   # h1_1 -> (h2_1, h2_2)
        [0.6,  0.7]    # h1_2 -> (h2_1, h2_2)
    ],
    # 第 3 層 (2→2)
    [
        [0.8,  0.0],   # h2_1 -> (h3_1, h3_2)
        [0.1, -0.6]    # h2_2 -> (h3_1, h3_2)
    ],
    # 第 4 層 (2→2) -- 輸出層
    [
        [0.5,  0.2],   # h3_1 -> (out1, out2)
        [-0.3, 0.9]    # h3_2 -> (out1, out2)
    ]
]
biases_3HL = [
    [ 0.01, -0.02],  # 第 1 層 (2 neurons)
    [ 0.3,   0.0  ], # 第 2 層 (2 neurons)
    [ 0.7,  -0.1  ], # 第 3 層 (2 neurons)
    [ 0.05,  0.15 ]  # 第 4 層 (2 neurons, 輸出)
]

# 用這些權重/偏置建構網路
nn_3hidden = Network(weights_3HL, biases_3HL)



# ---------------------------
# 測試
# ---------------------------
test_inputs1 = [(1.5, 0.5), (0, 1)]
for inputs in test_inputs1:
    outputs = nn1.forward(inputs)
    print(f"Network 1 - Input {inputs}: Output {outputs}")

test_inputs2 = [(0.75, 1.25), (-1, 0.5)]
for inputs in test_inputs2:
    outputs = nn2.forward(inputs)
    print(f"Network 2 - Input {inputs}: Output {outputs}")

# test_inputs3 = [
#     (1, 2, 3),
#     (-1, 0, 2)
# ]

# for inputs in test_inputs3:
#     out = nn3.forward(inputs)
#     print(f"Network 2 - Input {inputs}: Output {outputs}")
    

# test_inputs_3HL = [
#     (1, 0),
#     (2, 3),
#     (-1, 1),
# ]

# for inputs in test_inputs_3HL:
#     out = nn_3hidden.forward(inputs)
#     print(f"Network 2 - Input {inputs}: Output {outputs}")
