import timeit
import numpy as np
import logging
from datetime import datetime

from typing import Callable, Dict, Union, Tuple
from collections import namedtuple, OrderedDict

logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)
"""TimingProfile(iterations: int, number: int, warmup: int, duration: int, percentile: int or [int])"""
TimingProfile = namedtuple("TimingProfile", ["iterations", "number", "warmup", "duration", "percentile"])

def measure_python_inference_code(
    stmt: Union[Callable, str], timing_profile: TimingProfile
) -> None:
    """
    Measures the time it takes to run Pythonic inference code.
    Statement given should be the actual model inference like forward() in torch.

    Args:
        stmt (Union[Callable, str]): Callable or string for generating numbers.
        timing_profile (TimingProfile): The timing profile settings with the following fields.
            warmup (int): Number of iterations to run as warm-up before actual measurement cycles.
            number (int): Number of times to call function per iteration.
            iterations (int): Number of measurement cycles.
            duration (float): Minimal duration for measurement cycles.
            percentile (int or list of ints): key percentile number(s) for measurement.
    """

    warmup = timing_profile.warmup
    number = timing_profile.number
    iterations = timing_profile.iterations
    duration = timing_profile.duration
    percentile = timing_profile.percentile

    logging.debug(
        "Measuring inference call with warmup: {} and number: {} and iterations {} and duration {} secs".format(
            warmup, number, iterations, duration
        )
    )
    # Warmup
    warmup_mintime = timeit.repeat(stmt, number=number, repeat=warmup)
    logging.debug("Warmup times: {}".format(warmup_mintime))

    # Actual measurement cycles
    results = []
    start_time = datetime.now()
    iter_idx = 0
    while iter_idx < iterations or (datetime.now() - start_time).total_seconds() < duration:
        iter_idx += 1
        results.append(timeit.timeit(stmt, number=number))

    if isinstance(percentile, int):
        return np.percentile(results, percentile) / number
    else:
        return [np.percentile(results, p) / number for p in percentile]