import sys
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from loguru import logger  # loguru is a modern logging library.

from MLP import MLP
from scaler import Scaler1D, Scaler2D
from error import get_rmse


def sin_x_divide_x(x: float) -> float:
    if x == 0:  # Manually handle this to avoid 0 / 0
        return 1
    else:
        return np.sin(x) / x


if __name__ == '__main__':

    logger.remove()

    # Change the logging level here. Set level to 'DEBUG' may slow down the program.
    logger.add(sink=sys.stdout, level='INFO',
               format='<magenta>{time:HH:mm:ss}</magenta> | <level>{level: <9}</level> | <level>{message}</level>')

    logger.level('DEBUG', color='<yellow>')
    logger.level('INFO', color='<green>')

    random.seed(1)  # Keep the seed same to get the same value in each run.

    train_num_sqrt = 11  # 11 * 11 training samples.
    test_num_sqrt = 21  # 21 * 21 training samples.

    train_x = []

    for i in range(train_num_sqrt):
        for j in range(train_num_sqrt):
            train_x.append([i * 20 / (train_num_sqrt - 1) - 10,
                            j * 20 / (train_num_sqrt - 1) - 10])

    train_y = [sin_x_divide_x(x[0]) * sin_x_divide_x(x[1]) for x in train_x]

    plot_train_x = []

    for i in range(test_num_sqrt):
        for j in range(test_num_sqrt):
            plot_train_x.append([i * 20 / (test_num_sqrt - 1) - 10,
                                 j * 20 / (test_num_sqrt - 1) - 10])

    plot_train_y = [sin_x_divide_x(x[0]) * sin_x_divide_x(x[1])
                    for x in plot_train_x]

    plt.plot([i for i in range(test_num_sqrt * test_num_sqrt)], plot_train_y)

    # Scale the value to [0, 1]
    x_scaler = Scaler2D()
    y_scaler = Scaler1D()

    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    train_data = [[x, [y]] for x, y in zip(train_x, train_y)]

    # Set MLP network. No need to add input layer.
    mlp = MLP()
    # First hidden layer.
    mlp.add_layer(neuron_num=10, input_num=2, activation='ReLU')
    # Second hidden layer.
    mlp.add_layer(neuron_num=10, input_num=10, activation='ReLU')
    # Output layer.
    mlp.add_layer(neuron_num=1, input_num=10, activation='linear')

    # Train the network.
    epochs = 1000
    step = 0.05
    loss_list = mlp.train(train_data=train_data, epochs=epochs, step=step)

    predict_x = []

    for i in range(test_num_sqrt):
        for j in range(test_num_sqrt):
            predict_x.append([i * 20 / (test_num_sqrt - 1) - 10,
                              j * 20 / (test_num_sqrt - 1) - 10])

    true_predict_y = [sin_x_divide_x(x[0]) * sin_x_divide_x(x[1])
                      for x in predict_x]

    predict_x = x_scaler.transform(predict_x)
    # Use the trained network to predict.
    predict_y = [mlp.calc(x)[0] for x in predict_x]

    predict_x = x_scaler.inverse_transform(predict_x)
    predict_y = y_scaler.inverse_transform(predict_y)

    logger.info(f"Train RMSE: {loss_list[-1]}")
    logger.info(f"Test RMSE: {get_rmse(predict_y, true_predict_y)}")

    plt.plot([i for i in range(test_num_sqrt * test_num_sqrt)], predict_y)
    plt.savefig(
        f"./TEST2D/train_num{train_num_sqrt*train_num_sqrt}epochs{epochs}step{step}.png", dpi=600)

    plt.show()
    plt.close()

    plt.plot([i for i in range(len(loss_list))], loss_list)
    plt.savefig(
        f"./TEST2D/train_num{train_num_sqrt*train_num_sqrt}loss_epochs{epochs}step{step}.png", dpi=600)

    plt.show()
