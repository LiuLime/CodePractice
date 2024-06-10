# 递归
class Solution:
    def numDecodings(self, s: str) -> int:
        self.count = 0
        self.walk(0, s)
        return self.count

    def check_valid(self, i, n, s):
        flag = True
        string = s[i: i + n]
        if string.startswith("0"):
            flag = False
        num = int(string)
        if num > 26:
            flag = False
        return flag

    def walk(self, i, s):
        if i > len(s):
            return

        if i == len(s):
            self.count += 1
            return

        if i + 1 <= len(s) and self.check_valid(i, 1, s):
            self.walk(i + 1, s)
        else:
            return

        if i + 2 <= len(s) and self.check_valid(i, 2, s):
            self.walk(i + 2, s)
        else:
            return


class Solution2:
    def numDecodings(self, s: str) -> int:
        dp = [0 for _ in range(len(s))]

        if s[0] == "0":
            return 0
        if len(s) == 1:
            return 1
        if len(s) >= 2:
            dp[0] = 1
            dp[1] = 2
            if s[1] == "0":
                if int(s[0] + s[1]) <= 26:
                    dp[1] = 1
                else:
                    dp[1] = 0
            elif int(s[0] + s[1]) > 26:
                dp[1] = 1

            for i in range(2, len(s)):
                if s[i - 1] == "0" and s[i] == "0":
                    return 0
                elif s[i - 1] != "0" and s[i] == "0":
                    if int(s[i - 1] + s[i]) > 26:
                        return 0
                    else:
                        dp[i] = dp[i - 2]
                elif s[i - 1] == "0" and s[i] != "0":
                    dp[i] = dp[i - 1]
                elif int(s[i - 1] + s[i]) > 26:
                    dp[i] = dp[i - 1]
                else:
                    dp[i] = dp[i - 1] + dp[i - 2]
        return dp[len(s) - 1]


test = "27"
s = Solution()
print(s.numDecodings(test))

s2 = Solution2()
print(s2.numDecodings(test))
