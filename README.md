import numpy as np

# ==========================================
# 1. BASE LAYER INTERFACE
# ==========================================
class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError


# ==========================================
# 2. LAYER IMPLEMENTATIONS
# ==========================================
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # He (MSRA) Initialization to prevent gradient explosion/vanishing
        self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.params['b'] = np.zeros((1, output_size))
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, output_gradient):
        # Compute gradients with respect to parameters
        self.grads['W'] = np.dot(self.inputs.T, output_gradient)
        self.grads['b'] = np.sum(output_gradient, axis=0, keepdims=True)
        # Compute gradient with respect to inputs to pass backward
        return np.dot(output_gradient, self.params['W'].T)


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient):
        return output_gradient * (self.inputs > 0)


# ==========================================
# 3. OPTIMIZER COMPONENT
# ==========================================
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layers):
        for layer in layers:
            for key in layer.params:
                layer.params[key] -= self.lr * layer.grads[key]


# ==========================================
# 4. LOSS FUNCTION COMPONENT
# ==========================================
class MSELoss:
    @staticmethod
    def forward(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size


# ==========================================
# 5. MODEL CONTAINER PIPELINE
# ==========================================
class SequentialModel:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X, y, epochs, lr):
        optimizer = SGD(lr=lr)
        criterion = MSELoss()

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            loss = criterion.forward(predictions, y)

            # Backward pass
            loss_grad = criterion.backward(predictions, y)
            self.backward(loss_grad)

            # Parameter update
            optimizer.update(self.layers)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:4d}/{epochs} - Loss: {loss:.6f}")


# ==========================================
# 6. VERIFICATION & EXECUTION TEST
# ==========================================
if __name__ == "__main__":
    print("--- Testing Custom ML Framework ---")
    
    # Generate synthetic non-linear data (y = 2x^2 + 1)
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, (200, 1))
    y_train = 2 * (X_train ** 2) + 1 + np.random.normal(0, 0.1, (200, 1))

    # Construct the model pipeline
    model = SequentialModel()
    model.add(Dense(input_size=1, output_size=16))
    model.add(ReLU())
    model.add(Dense(input_size=16, output_size=1))

    # Train the custom model
    model.fit(X_train, y_train, epochs=1000, lr=0.01)
    
    # Run a test prediction
    test_input = np.array([[1.5]])
    predicted_val = model.forward(test_input)
    expected_val = 2 * (1.5 ** 2) + 1
    
    print("\n--- Evaluation ---")
    print(f"Input: {test_input[0][0]}")
    print(f"Predicted Output: {predicted_val[0][0]:.4f}")
    print(f"Expected Formula Output: {expected_val:.4f}")
    
