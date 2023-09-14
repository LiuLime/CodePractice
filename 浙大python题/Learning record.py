# 原字符串左侧对齐， 右侧补零，以零代替，若补全其他的，将0改为其他
ss = "52".ljust(6, '0')
print("ss:", ss)
# 原字符串右侧对齐， 左侧补零:
# 方法1
lr = "54".rjust(6, "0")
print("lr:", lr)
# 方法2
print('123'.zfill(6))
# 方法3
print('%06d' % 89)

# 向下取整
# a = 3.75
# int(a)
# 3
# 四舍五入
# round(3.25); round(4.85)
# 3.0
# 5.0
# 向上取整
# import math
# math.ceil(3.25)
# 4.0
# math.ceil(3.75)
# 4.0
# math.ceil(4.85)
# 5.0
# 分别取整数部分和小数部分
# import math
# math.modf(3.25)
# (0.25, 3.0)
# math.modf(3.75)
# (0.75, 3.0)
# math.modf(4.2)
# (0.20000000000000018, 4.0)
