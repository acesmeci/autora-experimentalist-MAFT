"""
Example Experimentalist
"""
import numpy as np
import pandas as pd

from typing import Union, List


def sample(
        conditions: Union[pd.DataFrame, np.ndarray], # essential
        models: List, 
        reference_conditions: Union[pd.DataFrame, np.ndarray], # already sampled conditions, see novelty conditions (maximum distance)
        num_samples: int = 1) -> pd.DataFrame: # essential 
    """
    Add a description of the sampler here.

    Args:
        conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        models: The sampler might use output from the theorist.
            Attention: `models` is a field of the standard state
        reference_conditions: The sampler might use reference conditons
        num_samples: number of experimental conditions to select

    Returns:
        Sampled pool of experimental conditions

    *Optional*
    Examples:
        These examples add documentation and also work as tests
        >>> example_sampler([1, 2, 3, 4])
        1
        >>> example_sampler(range(3, 10))
        3

    """

    if num_samples is None:
        num_samples = conditions.shape[0]

    # shape: n_samples | n_X's : Shape of matrix (X values vs amount of x's) pick rows to pick 


    random_indices = np.random.choice(list(range(len(conditions))), num_samples)
    new_conditions = conditions.iloc[random_indices]


    return new_conditions[:num_samples]
