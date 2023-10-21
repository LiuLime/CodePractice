# 7-27_BumbleArange

a = input("").split()
b = input("").split()
b = [int(i) for i in b]
c = b.copy()
N = int(a[0])
K = int(a[1])
index = list(range(N))

for i in range(0, K):
    for n in range(0, N-1):
        if c[n] > c[n+1]:
            index.insert(n, index[n+1])
            del index[n+2]
            # print('index:', index)
    c = [b[i] for i in index]
b = [str(b[i]) for i in index]
print(' '.join(b), end='')
