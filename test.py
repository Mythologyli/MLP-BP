import sys

from loguru import logger  # loguru is a modern logging library.

from MLP import MLP


if __name__ == '__main__':

    logger.remove()

    # Change the logging level here. Set level to 'DEBUG' may slow down the program.
    logger.add(sink=sys.stdout, level='INFO',
               format='<magenta>{time:HH:mm:ss}</magenta> | <level>{level: <9}</level> | <level>{message}</level>')

    logger.level('DEBUG', color='<yellow>')
    logger.level('INFO', color='<green>')

    mlp = MLP()
    mlp.add_layer(neuron_num=10, input_num=1, activation='sigmoid')
    mlp.add_layer(neuron_num=10, input_num=10, activation='sigmoid')
    mlp.add_layer(neuron_num=1, input_num=10, activation='linear')

    logger.info(f"Untrained network output: {mlp.calc([1])}")

    mlp.train(train_data=[[[1], [2]]], epochs=100, step=0.5)
    mlp.save_model('./Model/save.mlp')

    new_mlp = MLP()
    new_mlp.load_model('./Model/save.mlp')

    logger.info(f"Trained network output: {new_mlp.calc([1])}")
