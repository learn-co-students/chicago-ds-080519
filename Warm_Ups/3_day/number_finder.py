# let's import the things that we need
import numpy as np


def prime_finder(array):
    """ Finds all prime numbers in the given array """
    # we aim to use numpy vectorized operations to make this rapid.
    vf = np.vectorize(is_prime)
    prime_filter = vf(array)
    z = np.zeros(array.shape)
    return np.where(prime_filter, array, z).astype(np.int)


def is_prime(number):
    """ Determines whether the given number is a prime number.
        This serves as a helper function to prime_finder """
    if number <= 1:
        return False
    elif number == 2:
        return True
    elif number % 2 == 0:
        return False

    # note now that we only have to check half of the divisors of the number
    # and that the maximal divisor that we will have to check is no larger
    # than the square root. We take the floor as the divisor must be an
    # integer.
    highest_divisor = int(np.floor(np.sqrt(number)))

    # is there a faster way to do this?
    for i in range(2, 1 + highest_divisor):
        if number % i == 0:
            return False
    return True


def prime_finder_array(array):
    """ Determines whether each element of an array is a prime number """
    cond1 = array > 1  # primes must be larger than 1
    cond2 = array == 2  # 2 is a prime
    cond3 = (array % 2 != 0)  # but no other even number is a prime

    # this currently identifies all numbers which are either odd and larger
    # than 1, or are 2
    cond = np.logical_or(np.logical_and(cond1, cond3), cond2)

    # now we need to filter out the rest
    # we use a modified method as used above.
    highest_divs = np.floor(np.sqrt(array)).astype(np.int)

    # importantly, we should be able to simply do this for the maximal
    # highest divisor out of our array, as this will guarantee that we check
    # all the ones in between for each element of our original data.
    # this is a trade off that we make in this case, as not all of the
    # elements will need to be checked with such a high divisor
    # but this prevents us from needing to loop multiple times (so that
    # we may use more of NumPy's vectorization features).
    max_div = np.max(highest_divs)
    cond4 = np.ones(array.shape, dtype=bool)
    for i in range(2, max_div + 1):
        cond_part = array > i
        cond_part2 = array % i == 0
        # this is all things that are multiples of i and bigger than i
        cond_part3 = np.logical_and(cond_part, cond_part2)

        # but we actually want the opposite of that
        cond_part3 = np.logical_not(cond_part3)

        # now combine so that the overall condition of primeness gets
        # continually updated based on what has just been removed
        cond4 = np.logical_and(cond_part3, cond4)

    # put together the full conditions
    cond = np.logical_and(cond, cond4)

    z = np.zeros(array.shape)

    return np.where(cond, array, z).astype(np.int)


def fibonacci_finder(array):
    """ Finds all Fibonacci numbers in the given array """
    """ Finds all prime numbers in the given array """
    # we aim to use numpy vectorized operations to make this rapid.

    # note that a number n is fibonacci if and only if either one of
    # 5n^2 + 4 or 5n^2 - 4 is a perfect square
    case1_mat = (5 * (array**2) + 4).astype(np.int)
    case2_mat = (5 * (array**2) - 4).astype(np.int)

    # now we want to test squareness
    case1_isqrt = np.floor(np.sqrt(case1_mat)).astype(np.int)
    case2_isqrt = np.floor(np.sqrt(case2_mat)).astype(np.int)

    case1_square = (case1_isqrt**2).astype(np.int)
    case2_square = (case2_isqrt**2).astype(np.int)

    cond1 = np.equal(case1_mat, case1_square)
    cond2 = np.equal(case2_mat, case2_square)

    cond = np.logical_or(cond1, cond2)

    z = np.zeros(array.shape)

    return np.where(cond, array, z).astype(np.int)
