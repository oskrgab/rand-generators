from generators import randu, dessert_island
from scipy.stats import chisquare


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
    :return: True if we can't reject the null hypothesis of uniformity, False otherwise
    """

    _, p_value = chisquare(sample)

    return True if alpha < p_value < 1 - alpha else False


obs = [179, 208, 222, 199, 192]

a = chisquare(obs)
#%%
obs_2 = [190, 200, 150, 230]
a_2 = chisquare(obs_2)
