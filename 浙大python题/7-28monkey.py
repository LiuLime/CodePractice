# N = int(input())

#当一轮读完的时候，就自动index加上目前剩的个数的值，等于把前置位都挪到后面，

#一个静态index in个动态index

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
result = nums[::3]
print(result)  # 输出 [1, 4, 7, 10]
#[::3]表示从列表的第一个元素开始，每隔三个元素取一个数，最后得到的结果是包含被取出的元素的新列表result。

# 如果你希望从列表的第二个元素开始取数，可以将切片操作改为[1::3]。如果你希望从列表的最后一个元素开始取数，可以将切片操作改为[::-3]
i=0
new_nums = {}
for index in range(len(nums)):
    new_nums[index+1] = nums[index]

# print (new_nums)
nums_rm = nums[::3]
nums_left = list(set(nums)-set(nums_rm))

print(nums_left)
nums_rm = nums_left[::3]
nums_left = list(set(nums_left)-set(nums_rm))
print(nums_left)
nums_rm = nums_left[1::3]
nums_left = list(set(nums_left)-set(nums_rm))
print(nums_left)

N=3
for i in range (N):
    for element in nums:
        print(element)

        #可以循环每一个元素i次