import random
Game = """Rules: guess the number untill you correct, tips will be given every time you failed"""

print (Game)

R = random.randint(0,10)
# print(type(R))

a = input("Please input your int number between 0 to 10:",)

# if a.isdigit():
#    a= int(a)
# else:
#    a = input("Please input again:", )

while not a.isdigit():
    print("Please input again")
    a = input()

a = int(a)

while a!=R:
     if a <R:
        print("too small")
     if a >R:
        print("too big")
     a = input("input again:",)
     while not a.isdigit():
         print("Please input again")
         a = input()

     a = int(a)
     if a == R:
         # print ("sucess")
         break
     # if a.isdigit():
     #     a=int(a)

while a == R:
    print ("sucess")
    break









