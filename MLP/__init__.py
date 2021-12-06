import pickle

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

    def train(self, train_data: list, epochs: int, step: float) -> None:
        for i in range(epochs):
            for single_pair_train_data in train_data:
                layer_output_list = []

                # Calculate each layers' output.
                input = single_pair_train_data[0]
                for layer in self._layer_list:
                    input = layer.calc(input)
                    layer_output_list.append(input)

                output_desired = single_pair_train_data[1]
                output_layer_output_actual = layer_output_list[-1]

                layer_error_list = []  # Layer error in reverse order.

                # Calculate output layer error.
                if self._layer_list[-1].activation == 'sigmoid':
                    layer_error_list.append([(t - o) * o * (1 - o)
                                             for t, o
                                             in zip(output_desired, output_layer_output_actual)])
                else:
                    layer_error_list.append([t - o
                                             for t, o
                                             in zip(output_desired, output_layer_output_actual)])

                logger.info(
                    f"Epoch: {i + 1}/{epochs}; Error: {layer_error_list[-1]}")

                # Calculate all hidden layers' error.
                for i in range(len(layer_output_list) - 2, -1, -1):
                    # Calculate every neurons' error.
                    error = []
                    for j in range(len(self._layer_list[i].neuron_list)):
                        next_layer_error_sum = sum([next_layer_error * next_layer_neuron.weight_list[j]
                                                    for next_layer_error, next_layer_neuron
                                                    in zip(layer_error_list[-1], self._layer_list[i + 1].neuron_list)])

                        if self._layer_list[i].activation == 'sigmoid':
                            error.append(
                                layer_output_list[i][j] * (1 - layer_output_list[i][j]) * next_layer_error_sum)
                        else:
                            error.append(next_layer_error_sum)

                    layer_error_list.append(error)

                # Reverse the layer error list to right order.
                layer_error_list.reverse()

                # Update all layers.
                layer_output_with_input_list = [single_pair_train_data[0]] + \
                    layer_output_list

                for layer, layer_error, last_layer_output in zip(self._layer_list, layer_error_list, layer_output_with_input_list):
                    for neuron, e in zip(layer.neuron_list, layer_error):
                        neuron.weight_list = [w + step * e * o
                                              for w, o in zip(neuron.weight_list, last_layer_output)]

                        neuron.bias = neuron.bias + step * e

    def save_model(self, file: str) -> None:
        with open(file, "wb") as f:
            pickle.dump(self._layer_list, f)

    def load_model(self, file: str) -> None:
        with open(file, "rb") as f:
            self._layer_list = pickle.load(f)

    def __del__(self) -> None:
        logger.debug('Delete MLP.')
