from loguru import logger

from .neuron import Neuron


class Layer:

    def __init__(self, neuron_num: int, input_num: int, activation: str = 'sigmoid') -> None:
        self._neuron_list = []
        self._activation = activation

        for i in range(neuron_num):
            self._neuron_list.append(Neuron(input_num=input_num,
                                            activation=activation))

    @property
    def neuron_list(self) -> list:
        return self._neuron_list

    @neuron_list.setter
    def neuron_list(self, neuron_list) -> None:
        if len(neuron_list) != len(self._neuron_list):
            raise ValueError(
                f"Length of input neuron list {len(neuron_list)} doesn't fit length of existing neuron list {len(self._neuron_list)}")

        self._neuron_list = neuron_list

    @property
    def neuron_num(self) -> int:
        return len(self._neuron_list)

    @property
    def activation(self) -> str:
        return self._activation

    def calc(self, input: list) -> list:
        output = []
        for neuron in self._neuron_list:
            output.append(neuron.calc(input))

        logger.debug(f"Layer input: {input}, output: {output}.")

        return output

    def __del__(self) -> None:
        logger.debug('Delete layer.')
