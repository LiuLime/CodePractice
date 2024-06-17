# 231. Power of Two
# Easy

# 递归，要注意到n=0的情况
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        self.flag = False
        self.divide(n)
        return self.flag

    def divide(self, n):
        if n == 1:
            self.flag = True
        if n % 2 == 0 and n != 0:
            n = n // 2
            self.divide(n)

        return


"""别人的牛逼解法，通过二进制移位
假如n=8，那么二进制为1000，n-1为0111。`&`的意思是AND，所以如果n为2的幂次方，n&（n-1）==0"""


class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n != 0 and n & (n - 1) == 0


s = Solution()
s.isPowerOfTwo(0)
