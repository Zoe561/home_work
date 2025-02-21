import math
import random

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
    return sig_val * (1.0 - sig_val)

# =========== Loss Functions ===========
def bce_derivative(output, expect):
    epsilon = 1e-15
    o_i = min(max(output, epsilon), 1.0 - epsilon)
    e_i = expect
    return -( e_i / o_i - (1.0 - e_i)/(1.0 - o_i) )

def mse_derivative(output, expect):
    return 2.0 * (output - expect)

# =========== Neural Network Class ===========
class Network:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, 
                 hidden_activation=relu, output_activation='linear', 
                 w_h=None, w_o=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation_type = output_activation

        # # Initialize weights
        # self.weights_h = w_h if w_h else [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]]
        # self.weights_o = w_o if w_o else [[0.8, -0.5, 0.6]]

        # 使用 Xavier/Glorot 初始化
        if w_h is None:
            limit_h = math.sqrt(6 / (input_size + hidden_size))
            print(f"\n隱藏層權重初始化範圍: [{-limit_h:.4f}, {limit_h:.4f}]")
            self.weights_h = [[random.uniform(-limit_h, limit_h) for _ in range(input_size + 1)] 
                        for _ in range(hidden_size)]
            print("隱藏層初始權重值:")
            for i, row in enumerate(self.weights_h):
                print(f"神經元 {i + 1}: {[f'{w:.4f}' for w in row]}")
        else:
            self.weights_h = w_h

        if w_o is None:
            limit_o = math.sqrt(6 / (hidden_size + output_size))
            print(f"\n輸出層權重初始化範圍: [{-limit_o:.4f}, {limit_o:.4f}]")
            self.weights_o = [[random.uniform(-limit_o, limit_o) for _ in range(hidden_size + 1)]
                            for _ in range(output_size)]
            print("輸出層初始權重值:")
            for i, row in enumerate(self.weights_o):
                print(f"神經元 {i + 1}: {[f'{w:.4f}' for w in row]}")
        else:
            self.weights_o = w_o

        # Forward pass storage
        self.z_hidden = [0.0] * self.hidden_size
        self.a_hidden = [0.0] * self.hidden_size
        self.z_output = [0.0] * self.output_size
        self.a_output = [0.0] * self.output_size

        # Gradient storage
        self.grad_w_h = [[0.0]*(self.input_size+1) for _ in range(self.hidden_size)]
        self.grad_w_o = [[0.0]*(self.hidden_size+1) for _ in range(self.output_size)]

    def forward(self, x1, x2):
        for i in range(self.hidden_size):
            w1, w2, b = self.weights_h[i]
            z = w1*x1 + w2*x2 + b
            self.z_hidden[i] = z
            self.a_hidden[i] = relu(z) if self.hidden_activation == relu else z

        raw_outputs = []
        for i in range(self.output_size):
            w_h1, w_h2, b = self.weights_o[i]
            z = w_h1*self.a_hidden[0] + w_h2*self.a_hidden[1] + b
            self.z_output[i] = z
            raw_outputs.append(z)

        final_outputs = [linear(z) if self.output_activation_type == 'linear' else sigmoid_scalar(z) for z in raw_outputs]

        for i in range(self.output_size):
            self.a_output[i] = final_outputs[i]

        return final_outputs

    def forward_train(self, x1, x2):
        self.x1, self.x2 = x1, x2
        return self.forward(x1, x2)

    def backward(self, expects, loss_type='mse'):
        dL_dOut = [mse_derivative(self.a_output[i], expects[i]) if loss_type == 'mse' 
                   else bce_derivative(self.a_output[i], expects[i]) 
                   for i in range(self.output_size)]

        dL_dZ_out = [dL_dOut[i] * (linear_derivative(self.z_output[i]) if self.output_activation_type == 'linear' 
                                   else sigmoid_derivative(self.a_output[i])) for i in range(self.output_size)]

        for i in range(self.output_size):
            self.grad_w_o[i] = [dL_dZ_out[i] * self.a_hidden[j] for j in range(self.hidden_size)] + [dL_dZ_out[i]]

        dL_dA_hidden = [sum(dL_dZ_out[i] * self.weights_o[i][j] for i in range(self.output_size)) for j in range(self.hidden_size)]
        dL_dZ_hidden = [dL_dA_hidden[j] * (relu_derivative(self.z_hidden[j]) if self.hidden_activation == relu else 1) 
                        for j in range(self.hidden_size)]

        for j in range(self.hidden_size):
            self.grad_w_h[j] = [dL_dZ_hidden[j] * self.x1, dL_dZ_hidden[j] * self.x2, dL_dZ_hidden[j]]

    def zero_grad(self, learning_rate=0.01):
        for j in range(self.hidden_size):
            for k in range(self.input_size+1):
                self.weights_h[j][k] -= learning_rate * self.grad_w_h[j][k]
                self.grad_w_h[j][k] = 0.0
        for i in range(self.output_size):
            for k in range(self.hidden_size+1):
                self.weights_o[i][k] -= learning_rate * self.grad_w_o[i][k]
                self.grad_w_o[i][k] = 0.0
