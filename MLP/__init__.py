from loguru import logger

from .layer import Layer


class MLP:

    def __init__(self) -> None:
        self._layer_list = []

    def add_layer(self, neuron_num: int, input_num: int, activation: str = 'sigmoid') -> None:
        self._layer_list.append(Layer(neuron_num, input_num, activation))

    @property
    def layer_num(self) -> int:
        return len(self._layer_list)

    def calc(self, input: list) -> list:
        if isinstance(input, list) == False:
            raise TypeError(
                f"Input type {type(input)} doesn't fit required type list")

        output = input
        for layer in self._layer_list:
            output = layer.calc(output)

        logger.debug(f"MLP input: {input}, output: {output}.")

        return output

    def __del__(self) -> None:
        logger.debug('Delete MLP.')
