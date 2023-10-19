# 三整数倒序
# T = int(input())
# a = T//100
# bc = T % 100
# #print(a,bc)
# b = bc//10
# c = bc%10
# R = c*100+b*10+a
# print(R, end="")
# 字符串倒叙"-"为右往左读，1为读取长度
# T = input()
# a = T[::-1]
# a = int(a)
# print(a)

# #十进制转十六进制
# dec = int(input())
# hex = hex(dec)
# hex_d = hex[2:]
# print(hex_d,end="")

# 哪个小球不一样
# d = input()
# d = d.split( )
# print(d)
# A = d[0]
# B = d[1]
# C = d[2]
# if A==B:
#     print("C",end="")
# elif A==C:
#     print("B",end="")
# elif B==C:
#     print("A",end="")

# 几点钟
# start = input(' ')
# start2 = start.split(" ")
# start_time = start2[0]
# time = int(start2[1])
# start_min = int(start_time[-2:])
# # print(start_min)
# start_time = int(start2[0])
# start_hour = start_time - start_min
# # print(start_hour)
# judge = start_min + time
# # print(judge)
# if abs(judge) >= 60:
#     shang = judge//60
#     yu = judge % 60
# else:
#     shang = 0
#     yu = judge % 60
# end_min = yu
# end_hour = int(start_hour/100 + shang)
# # print(end_hour)
# print(end_hour, str(end_min).rjust(2,"0"), sep="", end="")

# 计算水费
# x = int(input())
# if x <= 15:
#     y = 4*x /3
# elif x > 15:
#     x2 = x-15
#     y = 4*x/3
#     y2 = 2.5*x2 - 17.5
#     y = y+y2
# print("%.2f"% y)

# 简单计算器
# import math
# a = input()
# a = a.split(" ")
# a1 = int(a[0])
# b = a[1]
# a2 = int(a[2])
# if b == "+":
#     result = a1 + a2
#     print(result,end="")
# elif b =="-":
#     result = a1 - a2
#     print(result, end="")
# elif b =="*":
#     result = a1 * a2
#     print(result, end="")
# elif b == "/":
#     result = a1/a2
#     print(math.trunc(result), end="")
# elif b == "%":
#     result = a1%a2
#     print(result, end="")
# else:
#     print ("ERROR",end="")

# import math
# CM = float(input())
# M = CM / 100/0.3048
# a = math.modf(M)
# # print(a)
# foot = a[1]
# inch = a[0]
# inch = int(inch*12)
# print (int(foot ),inch, end= "")

# start = input('') # print(' ',end='') start = input('') 一直报错的原因是input里加了个空格
# start2 = start.split(" ")
# start_time = start2[0]
# time = int(start2[1])
# start_min = int(start_time[-2:])
# start_time = int(start2[0])
# start_hour = start_time - start_min
# judge = start_min + time
# if abs(judge) >= 60:
#     shang = judge//60
#     yu = judge % 60
# else:
#     shang = 0
#     yu = judge % 60
# end_min = yu
# end_hour = int(start_hour/100 + shang)
# print("                      ")
# print(end_hour, str(end_min).rjust(2,"0"), sep="", end="")

# 通讯录问题
import random

d = {}
# l = []
d["name"] = {}
d["date"] = {}
d["phone"] = {}
d["telephone"] = {}
d["sexual"] = {}
number = int(input())

for i in range(0, number):
    record = input()
    record_list = record.split()
    # print(record_list)
    index = i

    d["name"][index] = record_list[0]
    d["date"][index] = record_list[1]
    d["sexual"][index] = record_list[2]
    d["phone"][index] = record_list[3]
    d["telephone"][index] = record_list[4]
# print(len(d["name"]), end="")

# Flag = True
K = input()
K = K.split(" ")
K_number = int(K[0])
l = K[1:]
for a in range(0, K_number):

    if int(l[a]) >= len(d["name"]):
        print("Not Found")
    else:
        b = int(l[a])
        if a == K_number-1:
            print(d["name"][b], d["phone"][b], d["telephone"][b], d["sexual"][b], d["date"][b],end='')
        else:
            print(d["name"][b], d["phone"][b], d["telephone"][b], d["sexual"][b], d["date"][b])

# 按空格取元素，按名字顺序排进字典 print的时候取下标print，
