import random

import numpy
from loguru import logger


class Neuron:

    def __init__(self, input_num: int, activation: str = 'linear') -> None:
        self._weight_list = []

        for i in range(input_num):
            self._weight_list.append(random.uniform(-1, 1))

        self._bias = random.uniform(-1, 1)
        self._activation = activation

        if self._activation not in {'linear', 'sigmoid', 'tanh'}:
            raise ValueError(f"Activation function {activation} not exist!")

        logger.debug(f"Create neuron, activation:{activation}.")

    @property
    def weight_list(self) -> float:
        return self._weight_list

    @weight_list.setter
    def weight_list(self, weight_list) -> None:
        if len(weight_list) != len(self._weight_list):
            raise ValueError(
                f"Length of input weight list {len(weight_list)} doesn't fit length of existing weight list {len(self._weight_list)}")

        self._weight_list = weight_list

    @property
    def input_num(self) -> float:
        return len(self._weight_list)

    def calc(self, input: list) -> float:
        if len(input) != len(self._weight_list):
            raise ValueError(
                f"Input num {len(input)} doesn't fit neuron input num {len(self._weight_list)}")

        output: float = 0
        for single_input, single_weight in zip(input, self._weight_list):
            output += single_input * single_weight

        output += self._bias

        if self._activation == 'sigmoid':
            output = 1 / (1 + numpy.exp(-output))
        elif self._activation == 'tanh':
            output = numpy.tanh(output)

        logger.debug(f"Neuron input: {input}, output: {output}.")

        return output

    def __del__(self) -> None:
        logger.debug('Delete neuron.')
