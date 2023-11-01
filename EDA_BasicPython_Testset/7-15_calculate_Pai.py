# 7-15  计算圆周率
import math


def the_an_item(n):
    """return an的值"""
    fenzi = math.factorial(n)
    fenmu = 1

    for i in range(1, n+1):
        fenmu = (2*i+1)*fenmu

    return fenzi/fenmu


shrefold = float(input(""))

if shrefold < 1:
    n = 0
    an_sum = 0
    while True:
        an = the_an_item(n)
        an_sum += an * 2
        if an > shrefold:
            n += 1
        else:
            print("%.6f"%an_sum,end='')
            break

