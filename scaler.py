class Scaler1D:

    def __init__(self) -> None:
        self.max_num: float = None
        self.min_num: float = None

    def transform(self, num_list: list) -> list:
        if self.max_num is None:
            self.max_num = max(num_list)

        if self.min_num is None:
            self.min_num = min(num_list)

        return [(x - self.min_num) / (self.max_num - self.min_num) for x in num_list]

    def inverse_transform(self, num_list: list) -> list:
        return [x * (self.max_num - self.min_num) + self.min_num for x in num_list]


class Scaler2D:

    def __init__(self) -> None:
        self.max_num: float = None
        self.min_num: float = None

    def transform(self, num_list: list) -> list:
        if self.max_num is None:
            self.max_num = max([max(x[0], x[1]) for x in num_list])

        if self.min_num is None:
            self.min_num = min([min(x[0], x[1]) for x in num_list])

        return [[(x[0] - self.min_num) / (self.max_num - self.min_num),
                 (x[1] - self.min_num) / (self.max_num - self.min_num)]
                for x in num_list]

    def inverse_transform(self, num_list: list) -> list:
        return [[x[0] * (self.max_num - self.min_num) + self.min_num,
                 x[1] * (self.max_num - self.min_num) + self.min_num]
                for x in num_list]
