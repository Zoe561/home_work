class Network:
    def __init__(self):
        self.W1 = [[0.5, 0.6], [1.5, -0.8]]
        self.b1_weights = [0.3, 1.25]
        self.W2 = [[0.6], [-0.8]]
        self.b2_weights = [0.3]
        self.W3 = [[0.5, -0.4]]
        self.b3_weights = [0.2, 0.5]

    def vector_add(self, v1, v2):
        return [a + b for a, b in zip(v1, v2)]

    def matrix_vector_multiply(self, matrix, vector):
        result = []
        for row in matrix:
            sum = 0
            for i in range(len(row)):
                sum += row[i] * vector[i]
            result.append(sum)
        return result

    def forward(self, inputs):
        h1h2 = self.vector_add(self.matrix_vector_multiply([self.W1[0], self.W1[1]], inputs), self.b1_weights)
        h3 = self.vector_add([h1h2[0] * self.W2[0][0] + h1h2[1] * self.W2[1][0]], self.b2_weights)
        outputs = self.vector_add([h3[0] * self.W3[0][0], h3[0] * self.W3[0][1]], self.b3_weights)
        return outputs

nn = Network()
test_inputs = [[0.75, 1.25], [-1, 0.5]]

for inputs in test_inputs:
    outputs = nn.forward(inputs)
    print(f"輸入 (X₁,X₂) = {tuple(inputs)} 的輸出結果:")
    print(f"O₁ = {outputs[0]:.4f}, O₂ = {outputs[1]:.4f}\n")
    
# TODO: 待優化