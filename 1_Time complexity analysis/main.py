import random
import matplotlib.pyplot as plt
import decimal
import numpy as np
from scipy.interpolate import interp1d
import timeit

decimal.getcontext().prec = 100
MIN_MERGE = 32

def is_even(myList, index):
    return myList[index] % 2 == 0

def sum_func(mylist):   #list is passed to the function
    summ = 0
    for n in mylist:
        summ += n
    return summ

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result


def polynom_direct(myList, x):
    poly_sum = decimal.Decimal(0)
    for k in range(len(myList)):
        poly_sum = decimal.Decimal(poly_sum) + decimal.Decimal(myList[k]) * decimal.Decimal((decimal.Decimal(x)**decimal.Decimal(k)))
    return decimal.Decimal(poly_sum)


def polynom_horner(myList, x):
    poly_sum = decimal.Decimal(0)
    for k in reversed(myList):
        poly_sum = poly_sum * decimal.Decimal(x) + decimal.Decimal(k)
    return decimal.Decimal(poly_sum)


def bubble_sort(our_list):
    for i in range(len(our_list)):
        for j in range(len(our_list) - 1):
            if our_list[j] > our_list[j+1]:
                our_list[j], our_list[j+1] = our_list[j+1], our_list[j]


def partition(arr, low, high):
    i = (low - 1)
    pivot = arr[high]
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)


def quickSort(arr, low, high):
    if len(arr) == 1:
        return arr
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


def calcMinRun(n):
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r


def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1


def merge(arr, l, m, r):
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])
    i, j, k = 0, 0, l
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1
    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1


def timSort(arr):
    n = len(arr)
    minRun = calcMinRun(n)
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)
    size = minRun
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge(arr, left, mid, right)
        size = 2 * size


def matrix_product(A, B, result):
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


N = list(range(1, 2001))

time_1 = [0] * len(N)
time_2 = [0] * len(N)
time_3 = [0] * len(N)
time_4_1 = [0] * len(N)
time_4_2 = [0] * len(N)
time_5 = [0] * len(N)
time_6 = [0] * len(N)
time_7 = [0] * len(N)
time_8 = [0] * len(N)

for n in N:
    v = [random.randrange(1, 100, 1) for r in range(n)]
    A = np.random.randint(10, size=(n, n))
    B = np.random.randint(10, size=(n, n))
    for k in range(1, 6):


        # point 1.1) - Constant function
        start = timeit.default_timer()
        temp = is_even(v, 0)
        time_1[n - 1] = time_1[n - 1] + (timeit.default_timer() - start) / 5

        
        
        # point 1.2) - the sum of elements v
        start = timeit.default_timer()
        temp = sum_func(v)
        time_2[n - 1] = time_2[n - 1] + (timeit.default_timer() - start) / 5
        #time_2[n - 1] = time_2[n - 1] + (time.time() - start) / 5

        
        
        # point 1.3) - the product of elements v
        start = timeit.default_timer()
        temp = multiplyList(v)
        time_3[n - 1] = time_3[n - 1] + (timeit.default_timer() - start) / 5

        
        
        # point 1.4.1) - the polynom for v direct method
        start = timeit.default_timer()
        temp = polynom_direct(v, 1.5)
        time_4_1[n - 1] = time_4_1[n - 1] + (timeit.default_timer() - start) / 5

        

        # point 1.4.2) - the polynom for v Horner's method
        start = timeit.default_timer()
        temp = polynom_horner(v, 1.5)
        time_4_2[n - 1] = time_4_2[n - 1] + (timeit.default_timer() - start) / 5

        
        
        # point 1.5) - the bubble sort of v
        v1 = v.copy()
        start = timeit.default_timer()
        bubble_sort(v1)
        time_5[n - 1] = time_5[n - 1] + (timeit.default_timer() - start) / 5



        # point 1.6) - the Quick sort of v
        v1 = v.copy()
        start = timeit.default_timer()
        quickSort(v1, 0, n - 1)
        time_6[n - 1] = time_6[n - 1] + (timeit.default_timer() - start) / 5

        
        
        # point 1.7) - the Timsort of v
        v1 = v.copy()
        start = timeit.default_timer()
        timSort(v1)
        time_7[n - 1] = time_7[n - 1] + (timeit.default_timer() - start) / 5


        #if n > 50:
        #   continue
        #point 2) - the product of matrices A and B
        result = np.zeros((n, n), dtype=int)
        start = time.time()
        temp = matrix_product(A, B, result)
        time_8[n - 1] = time_8[n - 1] + (time.time() - start) / 5



"""
# graph for point 1.1
plt.figure(1)
plt.plot(N, time_1)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for const function')



# graph for point 1.2
plt.figure(2)
#p1 = np.polyfit(N, time_2, 1)
#y_fit = np.polyval(p1, time_2)
#time_interp = interp1d(N, y_fit, kind='linear')
plt.plot(N, time_2)
#plt.legend(['data', 'linear interp'], loc='best')
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for sum of elements')



# graph for point 1.3
plt.figure(3)
plt.plot(N, time_3)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for product of elements')



# graph for point 1.4.1
plt.figure(4)
plt.plot(N, time_4_1)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for polynomial (direct method)')


# graph for point 1.4.2
plt.figure(5)
plt.plot(N, time_4_2)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for polynomial (Horner\'s method)')



# graph for point 1.5
plt.figure(6)
plt.plot(N, time_5)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for Bubble Sort')



# graph for point 1.6
plt.figure(7)
plt.plot(N, time_6)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for Quick Sort')



# graph for point 1.7
plt.figure(8)
plt.plot(N, time_7)
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for Timsort')



# graph for point 2
plt.figure(9)
plt.plot(N, time_8)
#plt.xlim([0, 50])
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for product of matrices')
"""

plt.plot(N, time_1, 'k-', N, time_2, 'r-', N, time_3, 'g-', N, time_4_1, 'b-', N, time_4_2, 'k--', N, time_5, 'r--', N, time_6, 'g--', N, time_7, 'b--', N, time_8, 'k-*-')
plt.xlabel('N')
plt.ylabel('time, s')
plt.title('Average execution time for every considered function')
plt.legend(['const', 'sum', 'prod', 'poly direct', 'poly Horner', 'Bubble', 'Quick', 'Tim', 'Matr Prod'], loc='best')

plt.show()

