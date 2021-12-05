from loguru import logger

from .neuron import Neuron


class Layer:

    def __init__(self, neuron_num: int, input_num: int, activation: str = 'sigmoid') -> None:
        self._neuron_list = []

        for i in range(neuron_num):
            self._neuron_list.append(Neuron(input_num=input_num,
                                            activation=activation))

    @property
    def neuron_num(self) -> int:
        return len(self._neuron_list)

    def calc(self, input: list) -> list:
        output = []
        for neuron in self._neuron_list:
            output.append(neuron.calc(input))

        logger.debug(f"Layer input: {input}, output: {output}.")

        return output

    def __del__(self) -> None:
        logger.debug('Delete layer.')
