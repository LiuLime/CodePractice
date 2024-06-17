# 415.Add strings 模拟题 Easy
# https://leetcode.com/problems/add-strings/description/

# 思路就是从最后一位开始逐个转变成数字相加再变成字符。
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        len1 = len(num1)
        len2 = len(num2)
        len_max = max(len1, len2)
        if len1 >= len2:
            num2 = "0" * (len1 - len2) + num2
        else:
            num1 = "0" * (len2 - len1) + num1
        sum = ["0"] * (len_max + 1)

        quotient = 0
        reminder = 0
        for i in range(-1, -len_max - 1, -1):
            single = int(num1[i]) + int(num2[i]) + quotient
            quotient = single // 10
            reminder = single % 10
            sum[i] = str(reminder)
            # print(sum)
            if quotient > 0:
                sum[i - 1] = str(quotient)
        sum_str = "".join(sum)
        if sum_str[0] == "0":
            sum_str = sum_str[1:]
        return sum_str


num1 = "11"
num2 = "123"
s = Solution()
s.addStrings(num1, num2)
