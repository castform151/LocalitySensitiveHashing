import random
from math import gcd
from time import time


def true_permutation(k):
    t0 = time()
    randList = []
    while k > 0:
        randIndex = random.randint(1, 499)

        # Ensure that the same value is not picked twice and GCD of Modulus and Coefficient is 1
        while (randIndex in randList or gcd(randIndex, 500) != 1):
            randIndex = random.randint(1, 499)

        randList.append(randIndex)
        k = k - 1
    print("Time taken to pick random coefficients: ", time() - t0)
    print(randList)
    return randList


true_permutation(20)
