import numpy as np
from activation import relu, sigmoid


def mse(y, y_received):
    return np.mean((y - y_received) ** 2)


def forward(data, network):
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


def backward():
    pass


def init_net(layers):
    np.random.seed(123)
    network = {}

    for layer_ind, layer in enumerate(layers[1:]):
        network[f"weight({layer_ind})"] = np.random.randn(layers[layer_ind], layer)
        network[f"bias({layer_ind})"] = np.zeros((1, layer))

    return network


def train_model(
    data_train, result_train, layers, learing_rate=0.01, epochs=10, batch_size=32
):
    network = init_net(layers)
    not_freq = epochs // 10

    for epoch in range(epochs):
        for i in range(0, data_train.shape[0], batch_size):
            data_batch = data_train[i : i + batch_size]
            result_batch = result_train[i : i + batch_size]

            res, cache_net = forward(data_batch, network)

        if epoch % not_freq == 0:
            result_current, cache_net = forward(data_train, network)
            training_loss = mse(result_train, result_current)
            print(f"Epoch: {epoch+1}, Loss: {training_loss}")
