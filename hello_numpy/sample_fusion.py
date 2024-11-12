import time

from line_profiler_pycharm import profile


def add(arr1, arr2):
    result_arr = []
    for i in range(len(arr1)):
        result_arr.append(arr1[i] + arr2[i])

    return result_arr


def multiple(arr1, arr2):
    result_arr = []
    for i in range(len(arr1)):
        result_arr.append(arr1[i] * arr2[i])

    return result_arr


def add_and_multiple(add_arr1, add_arr2, mul_arr):
    result_arr = []
    for i in range(len(add_arr1)):
        result_arr.append((add_arr1[i] + add_arr2[i]) * mul_arr[i])

    return result_arr


def main():
    arr1 = [1, 2, 3, 4, 5] * 100
    arr2 = [6, 7, 8, 9, 10] * 100
    arr3 = [2, 3, 4, 5, 6] * 100

    #  add_result = add(arr1, arr2)
    #  mul_result = multiple(add_result, arr3)
    fusion_operator_result = add_and_multiple(arr1, arr2, arr3)

    print(fusion_operator_result)


def main():
    arr1 = [1, 2, 3, 4, 5] * 100
    arr2 = [6, 7, 8, 9, 10] * 100
    arr3 = [2, 3, 4, 5, 6] * 100

    add_result = add(arr1, arr2)
    mul_result = multiple(add_result, arr3)

    s = time.process_time()
    fusion_operator_result = add_and_multiple(arr1, arr2, arr3)
    e = time.process_time()
    print(f"Fusion add_and_multiple 연산 수행시간:  {(e - s) * 1000:.3f}ms")

    print(mul_result)
    print(fusion_operator_result)


@profile
def main():
    arr1 = [1, 2, 3, 4, 5] * 1000
    arr2 = [6, 7, 8, 9, 10] * 1000
    arr3 = [2, 3, 4, 5, 6] * 1000

    s = time.process_time()
    add_result = add(arr1, arr2)
    mul_result = multiple(add_result, arr3)
    e = time.process_time()
    print(f"add & mul 연산 수행시간:  {(e - s) * 1000:.3f}ms")

    s = time.process_time()
    fusion_operator_result = add_and_multiple(arr1, arr2, arr3)
    e = time.process_time()
    print(f"Fusion add_and_multiple 연산 수행시간:  {(e - s) * 1000:.3f}ms")

    #print(mul_result)
    #print(fusion_operator_result)


if __name__ == '__main__':
    main()
