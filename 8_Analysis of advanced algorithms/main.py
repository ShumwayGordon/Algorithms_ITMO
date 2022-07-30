import math

import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import timeit

INT_MIN = -32767


def maxCrossingSum(arr, l, m, h):
    sm = 0
    left_sum = -10000
    for i in range(m, l - 1, -1):
        sm = sm + arr[i]

        if (sm > left_sum):
            left_sum = sm
    sm = 0
    right_sum = -1000
    for i in range(m + 1, h + 1):
        sm = sm + arr[i]

        if (sm > right_sum):
            right_sum = sm
    return max(left_sum + right_sum, left_sum, right_sum)


def maxSubArraySum(arr, l, h):
    if (l == h):
        return arr[l]
    m = (l + h) // 2
    return max(maxSubArraySum(arr, l, m),
               maxSubArraySum(arr, m + 1, h),
               maxCrossingSum(arr, l, m, h))


def cutRod_naive(price, n):
    if (n <= 0):
        return 0
    max_val = -sys.maxsize - 1

    # Recursively cut the rod in different pieces
    # and compare different configurations
    for i in range(0, n):
        max_val = max(max_val, price[i] +
                      cutRod_naive(price, n - i - 1))
    return max_val


def cutRod_dynamic(price, n):
    val = [0 for x in range(n + 1)]
    val[0] = 0

    for i in range(1, n + 1):
        max_val = INT_MIN
        for j in range(i):
            max_val = max(max_val, price[j] + val[i - j - 1])
        val[i] = max_val
    return val[n]


def maxSubArray_naive(arr):
  maximum = -math.inf
  for i in range(0, len(arr)):
    sum=0
    for j in range(i, len(arr)):
      sum += arr[j]
      maximum = max(sum, maximum) #compare the resulting sum with the existing maximum value
  return maximum


def main():

    # MAXIMUM SUBARRAY ALG

    # Driver Code
    arr = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    n = len(arr)

    max_sum = maxSubArraySum(arr, 0, n - 1)
    print("Maximum contiguous sum using D&C is ", max_sum)

    max_sum = maxSubArray_naive(arr)
    print("Maximum contiguous sum using Naive is ", max_sum)


    # CUTTING A ROD ALGORITM

    arr_2 = [1, 5, 8, 9, 10, 17, 17, 20]
    size = len(arr_2)
    print("Maximum Obtainable Value using Naive is", cutRod_naive(arr_2, size))

    print("Maximum Obtainable Value using Dynamic is", cutRod_dynamic(arr_2, size))

    N = list(range(1, 100))

    time_1 = [0] * len(N)
    time_2 = [0] * len(N)
    time_3 = [0] * len(N)
    time_4 = [0] * len(N)

    for n in N:
        v = [random.randrange(-50, 50, 1) for r in range(n)]
        v2 = [random.randrange(1, 100, 1) for r in range(n)]

        for k in range(6):

            start = timeit.default_timer()
            temp = maxSubArraySum(v, 0, n - 1)
            time_1[n - 1] = time_1[n - 1] + (timeit.default_timer() - start) / 6


            start = timeit.default_timer()
            temp = maxSubArray_naive(v)
            time_4[n - 1] = time_4[n - 1] + (timeit.default_timer() - start) / 6

            if n < 14:
                start = timeit.default_timer()
                temp = cutRod_naive(v2, n)
                time_2[n - 1] = time_2[n - 1] + (timeit.default_timer() - start) / 6


            else:
                time_2[n - 1] = time_2[12]

            start = timeit.default_timer()
            temp = cutRod_dynamic(v2, n)
            time_3[n - 1] = time_3[n - 1] + (timeit.default_timer() - start) / 6

    # graph for max subarray alg
    plt.figure(1)
    plt.plot(N, time_1)
    plt.xlabel('N')
    plt.ylabel('time, s')
    plt.title('Average execution time for Max Subarr Sum using D&C alg')

    # graph for cutting a Rod problem Naive approach
    plt.figure(2)
    plt.plot(N[0:13], time_2[0:13])
    plt.xlabel('N')
    plt.ylabel('time, s')
    plt.title('Average execution time for Cutting a Rog using naive approach')

    # graph for cutting a Rod problem Naive approach
    plt.figure(3)
    plt.plot(N, time_3)
    plt.xlabel('N')
    plt.ylabel('time, s')
    plt.title('Average execution time for Cutting a Rog using dynamic approach')

    N2 = [0] * len(N)
    print(len(N), len(N2))
    for i in range(len(N)-1):
        if i < 13:
            N2[i] = N[i]
        else:
            N2[i] = N[12]


    # graph for cutting a Rod problem Naive approach
    plt.figure(4)
    #plt.plot(N, time_2, N, time_3)
    plt.scatter(N2, time_2)
    plt.scatter(N, time_3)
    plt.xlabel('N')
    plt.ylabel('time, s')
    plt.title('Average execution time for Cutting a Rog naive & dynamic approaches')
    plt.legend(['naive', 'dynamic'], loc='best')

    # graph for cutting a Rod problem Naive approach
    plt.figure(5)
    # plt.plot(N, time_2, N, time_3)
    plt.scatter(N, time_1)
    plt.scatter(N, time_4)
    plt.xlabel('N')
    plt.ylabel('time, s')
    plt.title('Average execution time for Max Subarray sum naive & D&C')
    plt.legend(['D&C', 'naive'], loc='best')


    plt.show()



if __name__ == "__main__":
    main()






