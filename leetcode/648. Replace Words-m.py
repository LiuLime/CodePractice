# 648. Replace Words
# medium 我猜是贪心算法
from typing import List


class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        self.dictionary = dictionary
        self.sentence = sentence
        return

    def search_dictionary(self):
        # 实现按字母和长度排序
        self.dictionary = self.dictionary.sort()
        for word in sentence:
            for alpha in self.dictionary:
                if word[0] == alpha[0]:
                    self.word = word
                    self.alpha = alpha

    def match_word(self, word_idx, alpha_idx, word_, alpha_):
        if word_idx == len(word_) or alpha_idx == len(alpha_):
            return True

        if word_[word_idx] == alpha_[alpha_idx]:
            self.match_word(word_idx + 1, alpha_idx + 1, word_, alpha_)
        else:
            return False


dictionary = ["cat", "bat", "rat", "catt"]
sentence = "the cattle was rattled by the battery"
s = Solution()

test = s.match_word(0, 0, dictionary[0], sentence[1])
print(test)
