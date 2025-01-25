class Network:
    def __init__(self, network_type=1):
        self.type = network_type
        if network_type == 1:
            self.weights1 = [[0.5, 0.6], [0.2, -0.6]]
            self.bias1 = [0.3, 0.25]
            self.weights2 = [[0.8, 0.4]]
            self.bias2 = [-0.5]
        else:
            self.weights1 = [[0.5, 0.6], [1.5, -0.8]]
            self.bias1 = [0.3, 1.25]
            self.weights2 = [[0.6, -0.8], [0.5, 0.3]]
            self.bias2 = [0.2, 0.5]
            self.weights3 = [[-0.4, 0.5]]
            
    def forward(self, inputs):
        hidden = []
        for i in range(len(self.weights1[0])):
            val = 0
            for j in range(len(inputs)):
                val += inputs[j] * self.weights1[j][i]
            val += self.bias1[i]
            hidden.append(val)
            
        if self.type == 1:
            output = 0
            for h, w in zip(hidden, self.weights2[0]):
                output += h * w
            output += self.bias2[0]
            return [output]
        else:
            hidden2 = []
            for i in range(len(self.weights2)):
                val = 0
                for j in range(len(hidden)):
                    val += hidden[j] * self.weights2[i][j]
                hidden2.append(val)
            
            outputs = []
            for w in self.weights3[0]:
                out = 0
                for h in hidden2:
                    out += h * w
                outputs.append(out)
            return outputs

# 測試
nn1 = Network(1)
test_inputs1 = [(1.5, 0.5), (0, 1)]
for inputs in test_inputs1:
    outputs = nn1.forward(inputs)
    print(f"Network 1 - Input {inputs}: Output {outputs}")

nn2 = Network(2)
test_inputs2 = [(0.75, 1.25), (-1, 0.5)]
for inputs in test_inputs2:
    outputs = nn2.forward(inputs)
    print(f"Network 2 - Input {inputs}: Output {outputs}")