import numpy as np
from activation import relu, sigmoid, relu_dev
from sklearn.metrics import accuracy_score, f1_score


def mse(y, y_received):
    return np.mean((y - y_received) ** 2)


class Brain_network(object):
    def __init__(self, inp, outp, layr) -> None:
        self.inp = inp
        self.outp = outp
        self.layr = layr
        self.probed_epoch = []
        self.losses = []
        self.prints = {}


    def forward(self, data, network):
        new_net = {}
        input = data

        layers_cnt = (len(network) // 2) - 1

        for layer_ind in range(layers_cnt):
            output = (
                    np.dot(input, network[f"weight({layer_ind})"])
                    + network[f"bias({layer_ind})"]
            )
            input = relu(output)
            new_net[f"output({layer_ind})"] = output
            new_net[f"input({layer_ind})"] = input

        output = (
                np.dot(input, network[f"weight({layers_cnt})"]) + network[f"bias({layers_cnt})"]
        )
        input = sigmoid(output)
        new_net[f"output({layers_cnt})"] = output
        new_net[f"input({layers_cnt})"] = input

        return input, new_net

    def backward(self, data, result, network, cache_net):
        gradients = {}
        layers_cnt = len(network) // 2
        m = data.shape[0]

        # Ostatnia warstwa
        out_last = cache_net[f"input({layers_cnt - 1})"]
        dZ = out_last - result
        gradients[f"dweight({layers_cnt - 1})"] = (
                np.dot(cache_net[f"input({layers_cnt - 1})"].T, dZ) / m
        )
        gradients[f"dbias({layers_cnt - 1})"] = np.sum(dZ, axis=0, keepdims=True) / m

        # Warstwy ukryte
        for l in reversed(range(1, layers_cnt)):
            dA = np.dot(dZ, network[f"weight({l})"].T)
            dZ = dA * relu_dev(cache_net[f"output({l - 1})"])
            gradients[f"dweight({l - 1})"] = (
                    np.dot(data.T if l == 1 else cache_net[f"input({l - 2})"].T, dZ) / m
            )
            gradients[f"dbias({l - 1})"] = np.sum(dZ, axis=0, keepdims=True) / m

        return gradients

    def init_net(self, layers):
        np.random.seed(123)
        network = {}

        for layer_ind, layer in enumerate(layers[1:]):
            network[f"weight({layer_ind})"] = np.random.randn(layers[layer_ind], layer)
            network[f"bias({layer_ind})"] = np.zeros((1, layer))

        return network

    def update_network(self, network, gradients, learning_rate):
        layers_cnt = len(network) // 2 - 1

        for layer_ind in range(layers_cnt):
            network[f"weight({layer_ind})"] -= (
                    learning_rate * gradients[f"dweight({layer_ind})"]
            )
            network[f"bias({layer_ind})"] -= (
                    learning_rate * gradients[f"dbias({layer_ind})"]
            )

        return network

    def train_model(self,
                    data_train, result_train, layers, learning_rate=0.1, epochs=100, batch_size=64
                    , freq=10):
        network = self.init_net(layers)
        notification_freq = max(epochs // freq, 10)
        print_freq = epochs/10
        global epochs_g,batch_g,layers_g
        epochs_g = epochs
        batch_g = batch_size
        layers_g =layers
        count = -1
        for epoch in range(epochs):
            for i in range(0, data_train.shape[0], batch_size):
                data_batch = data_train[i: i + batch_size]
                result_batch = result_train[i: i + batch_size]

                res, cache_net = self.forward(data_batch, network)
                gradients = self.backward(data_batch, result_batch, network, cache_net)

                network = self.update_network(network, gradients, learning_rate)

            if (epoch + 1) % notification_freq == 0 or epoch == 0:
                result_current, cache_net = self.forward(data_train, network)
                training_loss = mse(result_train, result_current)
                self.probed_epoch.append(epoch + 1)
                self.losses.append(training_loss)

            if (epoch + 1) % print_freq == 0 or epoch == 0:
                count += 1
                result_current, cache_net = self.forward(data_train, network)
                training_loss = mse(result_train, result_current)
                self.prints[count] = {"key": (epoch+1),"value": training_loss}


        return network

    def evaluate_model(self, input_test, result_test, network):
        global epochs_g, batch_g,layers_g
        result_achived, _ = self.forward(input_test, network)
        result_pred = np.argmax(result_achived, axis=1)
        result_true = np.argmax(result_test, axis=1)

        f1 = f1_score(result_true, result_pred, average="micro")
        print(f"E:{epochs_g}, B:{batch_g}, L:{layers_g}")

        for values in self.prints.values():
            print("Epoch:", values['key'], "Loss:",values['value'])


        print(f"F1 Score: {f1}")
        return f1
