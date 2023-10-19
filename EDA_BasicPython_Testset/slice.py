# a = input()
# a = a[0:-1]
# a = a.split() #.split()的意思是以括号内的符号或空格来分割元素成列表，注意a需要是字符
# b = len(a)
# cnt = 0
# if b!= 0:
#   for i in a:
#       if cnt == b-1:
#           break
#       print (len(i),end=' ')
#       cnt += 1
#   print(len(a[b-1]))

l = [1,2,3,4,5,6]
l2 = l[::-1]
print(l2)
