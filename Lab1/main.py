import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.zeros(output_dim)
        self.grad_weights = None
        self.grad_biases = None
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x @ self.weights + self.biases

    def backward(self, grad_output):
        self.grad_weights = self.x.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weights.T
        return grad_input


class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.piecewise(
            x,
            [x < 0, x >= 0],
            [lambda x: np.exp(x) / (1 + np.exp(x)), lambda x: 1 / (1 + np.exp(-x))],
        )
        return self.output

    def backward(self, grad_output):
        return grad_output * np.multiply(self.output, 1 - self.output)


class Tanh(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, grad_output):
        return grad_output * (1 - np.tanh(self.x) ** 2)


class SoftSign(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return x / (1 + np.abs(x))

    def backward(self, grad_output):
        return grad_output / (1 + np.abs(self.x)) ** 2


class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output


class MSELoss(Layer):
    def forward(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def backward(self, y_pred, y):
        return 2 * (y_pred - y) / y.shape[0]


class SGD:
    def __init__(self, model, lr=1e-2):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                layer.weights -= self.lr * layer.grad_weights
                layer.biases -= self.lr * layer.grad_biases


class Adam:
    def __init__(self, model, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                self.m[layer] = np.zeros_like(layer.weights)
                self.v[layer] = np.zeros_like(layer.weights)

    def step(self):
        self.t += 1

        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                grad = layer.grad_weights

                self.m[layer] = self.beta1 * self.m[layer] + (1 - self.beta1) * grad

                self.v[layer] = self.beta2 * self.v[layer] + (
                    1 - self.beta2
                ) * np.square(grad)

                m_hat = self.m[layer] / (1 - self.beta1**self.t)

                v_hat = self.v[layer] / (1 - self.beta2**self.t)

                layer.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class Model:
    def __init__(self, optimizer="SGD", lr=1e-2, hidden_1=128, hidden_2=64, activation="Sigmoid"):
        if activation == "Sigmoid":
            activation = Sigmoid()
        elif activation == "Tanh":
            activation = Tanh()
        elif activation == "SoftSign":
            activation = SoftSign()

        layers = [
            Linear(2, hidden_1),
            activation,
            Linear(hidden_1, hidden_2),
            activation,
            Linear(hidden_2, 1),
            activation,
        ]
        self.nn = Sequential(layers)
        self.loss = MSELoss()
        if optimizer == "Adam":
            self.optimizer = Adam(self.nn, lr=lr)
        else:
            self.optimizer = SGD(self.nn, lr=lr)

    def backward(self, y, y_pred):
        grad_output = self.loss.backward(y_pred, y)
        return self.nn.backward(grad_output)

    def step(self):
        self.optimizer.step()

    def train(self, x, y, epochs=100000):
        records = {"loss": [], "accuracy": []}
        epoch = 0
        while (epochs > 0 and epoch < epochs) or (epochs == -1):
            epoch += 1
            y_pred = self.nn(x)

            acc = np.sum(np.where(y_pred > 0.5, 1, 0) == y) / len(y)
            if acc > 0.99:
                break

            loss = self.loss(y_pred, y)
            assert not np.isnan(loss)

            records["loss"].append(loss)
            records["accuracy"].append(acc)

            self.backward(y, y_pred)
            self.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch:7d}, Loss: {loss:2.6f}, Accuracy: {acc:.2f}")

        return records

    def test(self, x, y):
        res = []
        y_pred = self.nn(x)
        for i in range(x.shape[0]):
            res.append(f"Pred: {y_pred[i]}, True: {y[i]}")

        y_pred = np.where(y_pred > 0.5, 1, 0)
        print(f"Accuracy: {np.sum(y_pred == y) / len(y)}")
        return res


def generate_linear(n=100):
    import numpy as np

    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    import numpy as np

    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y, records, data, optimizer, lr, hidden_1, hidden_2, activation):
    import matplotlib.pyplot as plt

    y_hat = np.where(pred_y > 0.5, 1, 0)

    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize=18)
    for i in range(x.shape[0]):
        if y_hat[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.savefig(f"result_{data}_{optimizer}_{lr}_{hidden_1}_{hidden_2}_{activation}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(records["loss"], label="Loss", color="red")
    ax2.plot(records["accuracy"], label="Accuracy", color="blue")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.savefig(f"curve_{data}_{optimizer}_{lr}_{hidden_1}_{hidden_2}_{activation}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    acc = np.sum(y_hat == y) / len(y)
    plt.title(f"Accuracy: {acc:.2f}, loss: {MSELoss().forward(pred_y, y):.2f}", fontsize=18)
    
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Decision Boundary (y=0.5)')
    
    plt.scatter(range(len(y)), y, label="Ground Truth", color='black', marker="o", alpha=0.6, s=100)
    
    true_0_indices = (y == 0).flatten()
    true_1_indices = (y == 1).flatten()
    
    plt.scatter(np.where(true_0_indices)[0], pred_y[true_0_indices], 
                label="Pred (True=0)", color="red", marker="x", alpha=0.8, s=100)
    plt.scatter(np.where(true_1_indices)[0], pred_y[true_1_indices], 
                label="Pred (True=1)", color="blue", marker="x", alpha=0.8, s=100)
    
    for i, pred in enumerate(pred_y):
        color = "blue" if y[i] == 1 else "red"
        plt.annotate(f'{pred[0]:.2f}', 
                    (i, pred[0]),
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7,
                    rotation=45,
                    ha='left',
                    va='bottom',
                    color=color)
    
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"values_{data}_{optimizer}_{lr}_{hidden_1}_{hidden_2}_{activation}.png")
    plt.close()


def main(data="linear", optimizer="Adam", lr=1e-2, epochs=-1, hidden_1=32, hidden_2=16, activation="Sigmoid"):
    if data == "linear":
        x, y = generate_linear(100)
    elif data == "XOR":
        x, y = generate_XOR_easy()

    model = Model(optimizer=optimizer, lr=lr, hidden_1=hidden_1, hidden_2=hidden_2, activation=activation)
    records = model.train(x, y, epochs=epochs)
    pred_y = model.nn(x)
    show_result(x, y, pred_y, records, data, optimizer, lr, hidden_1, hidden_2, activation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="XOR")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--hidden_1", type=int, default=128)
    parser.add_argument("--hidden_2", type=int, default=64)
    parser.add_argument("--activation", type=str, default="Sigmoid")
    args = parser.parse_args()
    args = vars(args)
    main(**args)
