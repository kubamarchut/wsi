import numpy as np


def mse(y, y_received):
    return np.mean((y - y_received) ** 2)


def forward():
    pass


def backward():
    pass


def init_net(layers):
    np.random.seed(123)
    network = {}

    for layer_ind, layer in enumerate(layers):
        network[f"weight({layer_ind})"] = np.random.randn(layer)
        network[f"bias({layer_ind})"] = np.zeros((1, layer))

    return network


def train_model(
    input_train, output_train, layers, learing_rate=0.01, epochs=10, batch_size=32
):
    network = init_net(layers)
    print(network)
    not_freq = epochs // 10

    for epoch in range(epochs):

        if epoch % not_freq == 0:
            output_current = forward()
            # training_loss = mse(output_train, output_current)
            training_loss = 0
            print(f"Epoch: {epoch+1}, Loss: {training_loss}")
