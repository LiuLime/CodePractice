a = input('').split()

year = int(a[0])
hours = int(a[1])
if year < 5:
    salaryh = 30
    if hours <= 40:
        salary = salaryh*hours
    else:
        salary = 1200+salaryh*(hours-40)*1.5
else:
    salaryh = 50
    if hours <= 40:

        salary = salaryh*hours
    else:
        salary = 2000+salaryh*(hours-40)*1.5

print(format(salary, '.2f'),end='')

