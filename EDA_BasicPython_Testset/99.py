a = int(input())
cnt = 1
for i in range(1, a+1):
  for j in range (1, i+1):
    b = cnt * j
    print(j, "*", cnt, "=%-4d"%(b), sep="",end="")
  if j<a:
    print("")
  if cnt > a:
    break
  cnt = cnt + 1
