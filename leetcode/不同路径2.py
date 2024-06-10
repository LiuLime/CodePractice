"""https://leetcode.cn/problems/unique-paths-ii/description/"""
from typing import List


# 递归算法
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        self.obstacleGrid = obstacleGrid
        self.count = 0
        self.walk(0, 0, m - 1, n - 1)
        return self.count

    def check_stone(self, i, j):
        flag = True
        if self.obstacleGrid[i][j] == 1:
            flag = False
        return flag

    def walk(self, start_i, start_j, end_i, end_j):
        if not self.check_stone(start_i, start_j):
            return
        if start_i == end_i and start_j == end_j:
            self.count += 1

        if start_i + 1 <= end_i:
            self.walk(start_i + 1, start_j, end_i, end_j)
        if start_j + 1 <= end_j:
            self.walk(start_i, start_j + 1, end_i, end_j)


# 动态规划
class Solution2:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1:
            return 0
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        obstacleGrid.insert(0, [0] * m)
        for i in obstacleGrid:
            i.insert(0, 0)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[1][1] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if i == 1 and j == 1:
                    continue
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m][n]


obstacleGrid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
S = Solution2()
print(S.uniquePathsWithObstacles(obstacleGrid))
