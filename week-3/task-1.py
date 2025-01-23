import math

class Network:
    def __init__(self):
        self.weights1 = [
            [0.5, 0.6],
            [0.2, -0.6],
            [0.3, 0.25]
        ]
        self.weights2 = [
            [0.8],
            [0.4], 
            [-0.5]
        ]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def matrix_multiply(self, a, b):
        result = []
        for i in range(len(a)):
            current_row = []
            for j in range(len(b[0])):
                value = sum(a[i][k] * b[k][j] for k in range(len(b)))
                current_row.append(value)
            result.append(current_row)
        return result

    def forward(self, inputs):
        x = [inputs[0], inputs[1], 1]
        hidden_inputs = self.matrix_multiply([x], self.weights1)[0]
        hidden = [self.sigmoid(h) for h in hidden_inputs]
        hidden.append(1)
        output = self.matrix_multiply([hidden], self.weights2)[0]
        return self.sigmoid(output[0])

nn = Network()
outputs = nn.forward([1.5, 0.5])
print(outputs)

outputs = nn.forward([0, 1])
print(outputs)