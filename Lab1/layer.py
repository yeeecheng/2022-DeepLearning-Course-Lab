import numpy as np

class linear_layer:
    def __init__(self, in_features, out_features, bias= True):
        # y = Wx + b 
        k = 1.0 / in_features
        self.weight = np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_features, in_features))
        self.bias = np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_features))
        self.weight_grad = 0
        self.bias_grad = 0
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # input matrix W + bias
        self.a = x
        self.z = np.matmul(x, self.weight.T) + self.bias
        return self.z
    
    def update(self, lr= 1e-4):

        self.weight -= (lr * self.weight_grad).T
        self.bias -= lr * self.bias_grad

# activation function

class sigmoid:
    """ range[0, 1] """

    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)

    def derivative(self, x):
        return self(x) * (1 - self(x))

class tanh:
    """ range[-1, 1]"""

    def __init__(self):
        pass

    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)

    def derivative(self, x):
        return 1 - (self(x) ** 2)
        
