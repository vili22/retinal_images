import numpy as np


def random_search(n, *param_ranges):
    """
    Args:
      n (int):      Number of hyperparameter combinations to be generated.
      param_ranges: Each of the given arguments must be a list [`low`, `high`] where low
                     defines the `lower` and `high` defines the upper boundaries of the sampling interval
                     for the corresponding parameter.
    Returns:
      An iterator over n combinations of the hyperparameters. Each hyperparameter value is drawn uniformly
      from interval [low, high] specified by the corresponding input argument.
    """
    combinations = []
    for combination_iter in range(0, n):
        new_combination = []
        for param_range in param_ranges:
            new_combination.append(int(np.random.uniform(param_range[0], param_range[1])))

        combinations.append(tuple(new_combination))
    return iter(combinations)