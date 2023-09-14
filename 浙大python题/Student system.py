# for i in range input
# ...input()
# ...input()
# number
# name id          loop number
# dict
# print dict {name1:id1,name2:id2 ... etc }

# L = [] #建立list
# L.append(i) #往list里添加元素
# print(L)

# d = {} #建立空字典
# d[lyt] = {} #添加lyt key并指定其为一个字典

# lyt = 'gvjgvhjvhj'
# d = {}
# d[lyt] = {}
#
#
# d = {}
# d[4] = {}
d = {}
while True:

    a = input("Please input student number:", )
    if not a.isdigit():
        print("error,click return")
    else:
        a = int(a)
        break

for i in range(0, a):
    print("student", i + 1, ":")
    student = input()

    print("student_id", i + 1, ":")
    student_id = input()
    while student_id in d:
        print(d.get(student_id), ":", student_id, "repeat")
        student_id = input("input student_id again: ", )
    d[student_id] = student

for student_id in d:
    print(d[student_id], student_id)

