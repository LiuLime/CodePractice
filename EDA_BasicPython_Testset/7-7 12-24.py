a = input('').split(':')

h = int(a[0])
m = int(a[1])
if h < 12:
    a = str(h) + ':' + str(m)
    print(a,'AM',end='')
elif h == 12:
    a = str(h) + ':' + str(m)
    print(a,'PM',end='')

else:
    h = h - 12
    a = str(h)+':'+str(m)
    print(a, 'PM', end='')