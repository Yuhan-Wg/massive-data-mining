"""
Hash Functions
"""
from numbers import Number
from typing import FrozenSet

__all__ = [
    "hash2vector", "hash2int"
]

INT32_MAX = 0xFFFFFFFF
a, b = 1103515245, 12345


def lcg_hash(seed, min_value, max_value):
    return int((max_value - min_value) * (seed / INT32_MAX)) + min_value


def hash2vector(obj, length, min_value=0, max_value=0xFFFFFFFF, seed=0):
    """
    This function hashes string/integer to a vector with Linear Congruential Generator.
    :param obj: str/int
        hasattr(o, "__str__") = True
    :param length: int
        The length of returned vector.
    :param min_value: int
        The minimum value to produce.
    :param max_value:
        The maximum value to produce.
    :param seed: int
        The random seed to determine the hashing results.
    :return: Tuple<int>
    """
    if isinstance(obj, Number):
        seed = (seed + int(obj) * 31) & INT32_MAX
    elif isinstance(obj, str):
        seed = str2int(obj, seed)
    elif isinstance(obj, FrozenSet):
        for item in obj:
            seed += (int(item) if isinstance(item, Number)
                     else str2int(item, seed))
            seed &= INT32_MAX
    else:
        raise ValueError("The hashed object type {0} doesn't match the required types.".format(type(obj)))

    for _ in range(length):
        seed = (a * seed + b) & INT32_MAX
        yield lcg_hash(seed, min_value, max_value)


def hash2int(obj, min_value=0, max_value=INT32_MAX, seed=0):
    if isinstance(obj, Number):
        seed = (seed + int(obj) * 31) & INT32_MAX
    elif isinstance(obj, str):
        for character in obj:
            seed = (seed * 31 + ord(character)) & INT32_MAX
    else:
        raise ValueError("The hashed object type {0} doesn't match the required types.".format(type(obj)))
    seed = (a * seed + b) & INT32_MAX
    return lcg_hash(seed, min_value, max_value)


def str2int(obj: str, seed) -> int:
    for character in obj:
        seed = (seed * 31 + ord(character)) & INT32_MAX
    return seed








