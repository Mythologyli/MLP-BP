import numpy


def get_rmse(predict_list, true_list) -> float:
    if len(predict_list) != len(true_list):
        raise ValueError(
            f"Length of predict list {len(predict_list)} doesn't fit length of true list {len(true_list)}")

    sum = 0
    for predict, true in zip(predict_list, true_list):
        sum += (predict - true) ** 2

    return numpy.sqrt(sum / len(predict_list))
