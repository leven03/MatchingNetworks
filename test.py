import pickle



# with open("./omniglot_dataset.pkl", "rb") as f:
#     omniglot_dataset = pickle.load(f)
# print(omniglot_dataset)


import torch

a = torch.randn(2, 2, 2, 2)

print(a)
print(a.view(2, -1, 2))



# import concurrent.futures
# import math
# import time
#
#
# PRIMES = [
#     112272535095293,
#     112582705942171,
#     112272535095293,
#     115280095190773,
#     115797848077099,
#     115797848077098,
#     1099726899285419,
#     1122725350545293,
#     112582705542171,
#     112272554095293,
#     115280063190773,
#     115797886077099,
#     1157977656077098,
#     1099726835285419,
# ]
#
#
# def is_prime(n):
#     """
#     to judge the input number is prime or not
#     :param n: input number
#     :return: True or False
#     """
#     if n % 2 == 0:
#         return False
#
#     sqrt_n = int((math.sqrt(n)))
#     for i in range(3,sqrt_n + 1, 2):
#         if n % i == 0:
#             return False
#         return True
#
#
# def main():
#     """
#     create Process Pool to judge the numbers is prime or not
#     :return: None
#     """
#     start_time = time.time()
#     with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
#         for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
#             print(number,prime)
#     print("\nmax_workers = 2, time consume: %f" % ((time.time() - start_time) * 1000))
#
#     start_time = time.time()
#     with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#         for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
#             print(number,prime)
#     print("\nmax_workers = 4, time consume: %f" % ((time.time() - start_time) * 1000))
#
#     start_time = time.time()
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
#             print(number, prime)
#     print("\nmax_workers = None, time consume: %f" % ((time.time() - start_time) * 1000))
#
# if __name__ == '__main__':
#     main()
