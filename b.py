def binary_search_nearest(arr, target):
    left, right = 0, len(arr) - 1
    nearest = None

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return arr[mid]
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

        if nearest is None or abs(arr[mid] - target) < abs(nearest - target):
            nearest = arr[mid]

    return nearest


print(binary_search_nearest([1, 2], 5))
