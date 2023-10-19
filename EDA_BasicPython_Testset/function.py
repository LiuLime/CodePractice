import math




# flag = True
# result = 0
# if b == "+":
#     result = a1 + a2
#     #print(result,end="")
# elif b =="-":
#     result = a1 - a2
#     #print(result, end="")
# elif b =="*":
#     result = a1 * a2
#     #print(result, end="")
# elif b == "/":
#     result = math.trunc(a1/a2)
#     #print(result, end="")
# elif b == "%":
#     result = a1%a2
#     #print(result, end="")
# else:
#     print ("ERROR",end="")
#     flag = False
# if flag == True:
#     print(result, end="")

# def caculate(a1, b='+', a2=0):
#     flag = True
#     result = 0
#     if b == "+":
#         result = a1 + a2
#         # print(result,end="")
#     elif b == "-":
#         result = a1 - a2
#         # print(result, end="")
#     elif b == "*":
#         result = a1 * a2
#         # print(result, end="")
#     elif b == "/":
#         result = math.trunc(a1 / a2)
#         # print(result, end="")
#     elif b == "%":
#         result = a1 % a2
#         # print(result, end="")
#     else:
#         flag = False
#     return result, flag
#
#
#
# cnt = 2
# for i in range(cnt):
#     a = input()
#     a = a.split(" ")
#     a11 = int(a[0])
#     bb = a[1]
#     a22 = int(a[2])
#
#     ans, f = caculate(a1=a11, a2=a22)
#
#
#     if f == True:
#         print(ans, end="")
#     else:
#         print("ERROR", end="")


def add(ll,number):
    ll.append(number)
    print(ll)


# l = [1,2,3]
# add(l.copy(),4)
# print(l)

# def plus(number):
#     number += 1
#     print(number)
#
# n = 3
# plus(n)
# print(n)