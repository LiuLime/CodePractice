# easy level
# 1137. Nth-tribonacci
# 24/5/30 11:27

# recussion
class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        res = self.tribonacci(n - 1) + self.tribonacci(n - 2) + self.tribonacci(n - 3)
        return res


# dynamic programming
class Solution2:
    def tribonacci(self, n: int) -> int:
        self.n = n
        return self.walk()

    def walk(self, ):
        dp = [0 for _ in range(self.n + 1)]
        for i in range(0, self.n + 1):
            if i == 0:
                dp[i] = 0
                continue
            if i == 1 or i == 2:
                dp[i] = 1
                continue
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
        return dp[-1]


s2 = Solution2()
print(s2.tribonacci(4))
