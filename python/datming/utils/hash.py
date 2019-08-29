"""
Hash Functions
"""


__all__ = [
    "hash2vector"
]

INT32_MAX = 2 ** 32 - 1
a, c = 1103515245, 12345


def lcg(seed, min_value, max_value, n):
    for _ in n:
        seed = (a * seed + c) & INT32_MAX
        yield int((max_value - min_value) * (seed / INT32_MAX)) + min_value


def hash2vector(obj, length, min_value=0, max_value=INT32_MAX, seed=0):
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
    if isinstance(obj, int):
        seed = (seed + obj * 31) & INT32_MAX
    elif isinstance(obj, str):
        for c in obj:
            seed = (seed * 31 + ord(c)) & INT32_MAX
    else:
        raise ValueError("The object to be hashed doesn't match the required types.")

    return lcg(seed, min_value, max_value, length)










