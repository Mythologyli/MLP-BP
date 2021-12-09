import sys
import random

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger  # loguru is a modern logging library.

from MLP import MLP
from scaler import Scaler1D
from error import get_rmse


if __name__ == '__main__':

    logger.remove()

    # Change the logging level here. Set level to 'DEBUG' may slow down the program.
    logger.add(sink=sys.stdout, level='INFO',
               format='<magenta>{time:HH:mm:ss}</magenta> | <level>{level: <9}</level> | <level>{message}</level>')

    logger.level('DEBUG', color='<yellow>')
    logger.level('INFO', color='<green>')

    random.seed(50)  # Keep the seed same to get the same value in each run.

    train_num = 10  # 10 training samples.
    test_num = 361  # 361 testing samples.

    train_x = [i * 2 * np.pi / (train_num - 1) for i in range(0, train_num)]
    train_y = [np.sin(x) * np.cos(x) for x in train_x]

    plt.plot(train_x, train_y)

    # Scale the value to [0, 1]
    x_scaler = Scaler1D()
    y_scaler = Scaler1D()

    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    train_data = [[[x], [y]] for x, y in zip(train_x, train_y)]

    # Set MLP network. No need to add input layer.
    mlp = MLP()
    # First hidden layer.
    mlp.add_layer(neuron_num=10, input_num=1, activation='ReLU')
    # Second hidden layer.
    mlp.add_layer(neuron_num=10, input_num=10, activation='ReLU')
    # Output layer.
    mlp.add_layer(neuron_num=1, input_num=10, activation='linear')

    # Train the network.
    epochs = 1000
    step = 0.05
    loss_list = mlp.train(train_data=train_data, epochs=epochs, step=step)

    predict_x = [i * 2 * np.pi / (test_num - 1) for i in range(0, test_num)]
    true_predict_y = [np.sin(x) * np.cos(x) for x in predict_x]

    predict_x = x_scaler.transform(predict_x)
    # Use the trained network to predict.
    predict_y = [mlp.calc([x])[0] for x in predict_x]

    predict_x = x_scaler.inverse_transform(predict_x)
    predict_y = y_scaler.inverse_transform(predict_y)

    logger.info(f"Train RMSE: {loss_list[-1]}")
    logger.info(f"Test RMSE: {get_rmse(predict_y, true_predict_y)}")

    plt.plot(predict_x, predict_y)
    plt.savefig(f"./TEST1D/train_num{train_num}epochs{epochs}step{step}.png", dpi=600)

    plt.show()
    plt.close()

    plt.plot([i for i in range(len(loss_list))], loss_list)
    plt.savefig(f"./TEST1D/train_num{train_num}loss_epochs{epochs}step{step}.png", dpi=600)

    plt.show()
