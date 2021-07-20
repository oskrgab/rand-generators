import copy
import numpy as np


def lcg(seed, a, c, m, size=1):
    assert seed > 0
    x = np.zeros(size)
    x[0] = copy.copy(seed)

    a_f = float(a)
    c_f = float(c)
    m_f = float(m)

    if size > 1:
        for i in range(1, size):
            x[i] = (a_f * x[i-1] + c_f) % m_f

        return x / m_f
    else:
        return x[0] / m_f


def randu(seed, size=1):
    """
    The RANDU generator

    :param seed: An odd number between 1 and 2^31
    :param size: Determines the size of the sample of PRN's
    :return: Returns a numpy array with "size" length

    """

    a = 65539
    c = 0
    m = 2 ** 31

    return lcg(seed, a, c, m, size)


def dessert_island(seed, size=1):
    """
    The "dessert island" generator which is an "ok" generator

    :param seed: Any integer between 1 and 2^31 - 1
    :param size: Determines the size of the sample of PRN's
    :return: Returns a numpy array with "size" length

    """
    a = 16807
    c = 0
    m = 2 ** 31 - 1

    return lcg(seed, a, c, m, size)

