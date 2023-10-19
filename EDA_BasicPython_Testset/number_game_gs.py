import random

Game = """Rules: guess the number untill you correct, tips will be given every time you failed"""

print (Game)

R = random.randint(0,10)

while True:
    
    a = input("Please input your int number between 0 to 10:",)
    while not a.isdigit():
        print("Please input again")
        a = input()
    a = int(a)

    if a>R:
        print("too big")
    elif a<R:
        print("too small")
    else:
        break

print ("sucess")