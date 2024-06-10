# easy level
# 509. Fibonacci Number

# recurssion
class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        res = self.fib(n - 1) + self.fib(n - 2)
        return res


# dynamic 12:28
class Solution2:
    def fib(self, n: int) -> int:
        dp = [0 for _ in range(n + 1)]
        for i in range(0, n + 1):
            if i == 0:
                dp[i] = 0
            if i == 1:
                dp[i] = 1
            if i >= 2:
                dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

s=Solution2()
print(s.fib(2))