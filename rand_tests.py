from scipy.stats import chisquare, norm, kstest
import numpy as np


def gof_test(sample, alpha=0.05, bins=256):
    """
    Use chi-square to create goodness of fit test for pseudo random numbers (test for
    uniformity) where:

    - H0 (null hypothesis) is that the sample is uniform
    - H1 (alternate hypothesis) the sample is not uniform

    To test for uniformity, the p-value should lie in the range of alpha < p < 1 - alpha

    :param sample: Generated samples from random generators
    :param alpha: Significance level, default of 0.05
    :param bins: The number of bins to divide the samples
    :return: True if we can't reject the null hypothesis of uniformity, False otherwise.
    Also the p-value of the statistic wih a chi-square distribution
    """

    intervals = np.linspace(0, 1, bins)
    observations, _ = np.histogram(sample, intervals)

    _, p_value = chisquare(observations)

    return True if alpha < p_value < 1 - alpha else False, p_value


def run_test_up_down(sample, alpha=0.05):
    """
    Run test for up and down.

    :param sample: Generated samples from random generators
    :param alpha: Significance level, default of 0.05
    :return: True if we fail to reject the null hypothesis of independence. Also returns
    the z0 statistic.
    """

    a = 1  # Initialize to one since there will always be at least one run
    up = True if sample[1] >= sample[0] else False  # First up, down comparison

    # Counting all the ups and downs
    for i, x in enumerate(sample[1:-1], start=1):
        next_up = True if sample[i + 1] >= x else False
        if next_up != up:
            a += 1
            up = next_up

    n = len(sample)
    mu_a = (2 * n - 1) / 3
    var_a = (16 * n - 29) / 90

    z0 = (a - mu_a) / np.sqrt(var_a)
    z = norm.ppf(1 - alpha / 2)

    return True if abs(z0) < z else False, z0


def ks_test(sample, alpha=0.05):
    """
    Kolmogorov-Smirnov test. It is basically the same implementation as the one in scipy
    with the added difference that a boolean is returned on whether the null hypothesis
    of uniformity is not rejected (True) and false otherwise.

    :param sample: Generated samples from random generators
    :param alpha: Significance level, default of 0.05
    :return: True if we fail to reject the null hypothesis of uniformity. Also returns
    the p-value corresponding to the Kolmogorov-Smirnov test.
    """

    d0, p_value = kstest(sample, "uniform")

    return True if alpha < p_value else False, p_value


def correlation_test(sample, alpha=0.05):
    """
    Correlation test using class notes

    :param sample: Generated samples from random generators
    :param alpha: Significance level, default of 0.05
    :return: True if we fail to reject the null hypothesis of independence, False
    otherwise.
    """

    rho = 0
    n = len(sample)

    for k, rk in enumerate(sample[:-1]):
        rho += rk * sample[k + 1]

    rho = (12 / (n - 1)) * rho - 3
    var_rho = (13 * n - 19) / (n - 1) ** 2

    z0 = rho / np.sqrt(var_rho)
    z = norm.ppf(1 - alpha / 2)

    return True if abs(z0) < z else False, z0
