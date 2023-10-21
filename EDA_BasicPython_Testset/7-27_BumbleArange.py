# 7-27_BumbleArange

import time

a = input("").split()
N = int(a[0])
K = int(a[1])

# b = input("").split()
# b = [int(i) for i in b]

# test speed
# 方法1，没有减去不需要扫描的末尾项
b = list(range(N))
d = b.copy()

start = time.time()
for i in range(0, K):
    for n in range(0, N-1):
        if b[n] > b[n + 1]:
            b.insert(n, b[n + 1])
            del b[n + 2]
end = time.time()
# print(' '.join(str(i) for i in b), end='')
print(f"耗时{end - start}s")

# 方法2，减去不需要扫描的末尾项
start = time.time()
for i in range(0, K):
    itr = N - i - 1
    for n in range(0, itr):
        if d[n] > d[n + 1]:
            d.insert(n, d[n + 1])
            del d[n + 2]
end = time.time()
# print(' '.join(str(i) for i in d), end='')
print(f"耗时{end-start}s")

# 不需要的复杂代码
# c = b.copy()
# index = list(range(N))
#
# for i in range(0, K):
#     for n in range(0, N-1):
#         if c[n] > c[n+1]:
#             index.insert(n, index[n+1])
#             del index[n+2]
#             # print('index:', index)
#     c = [b[i] for i in index]
# b = [str(b[i]) for i in index]
# print(' '.join(b), end='')