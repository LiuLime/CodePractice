"""https://leetcode.cn/problems/unique-paths/"""


# 递归算法
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        self.count = 0
        self.walk(0, 0, m - 1, n - 1)
        return self.count

    def walk(self, start_i, start_j, end_i, end_j):
        if start_i == end_i and start_j == end_j:
            self.count += 1

        if start_i + 1 <= end_i:
            self.walk(start_i + 1, start_j, end_i, end_j)
        if start_j + 1 <= end_j:
            self.walk(start_i, start_j + 1, end_i, end_j)


# 动态规划
class Solution2:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[1][1]=1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if i==1 and j==1:
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m][n]


S = Solution()
print(S.uniquePaths(3, 2))
S2 = Solution2()
print(S2.uniquePaths(3, 2))
