"""
注册成功返回true，失败返回false，同一个用户名不能反复注册。用户名或密码为空也会注册失败。
"""
registered = {}
loggedIn = {}


def register(username, password) -> dict:
    global registered
    if username.isspace() or password.isspace() is True:
        return False
        # print("Blank error")

    elif username in registered:
        return False
        # print("false")

    elif username not in registered:
        # print ("Start to register")
        registered[username] = password
        return True
        # print(registered, "registered successfully")


""" 
登录成功返回true，失败返回false，同一个用户反复登录会失败(例如已经登录过了，还没有登出就
再次登录会失败)，没有注册过的用户登录也会失败，登录密码错误的也会失败。
"""


def login(username, password) -> bool:
    global loggedIn, registered

    if username not in registered:
        return False
        # print("false,not registered")
    elif username in loggedIn:
        return False
        # print("false, repeated log in")
    elif username not in loggedIn:
        password_true = registered[username]
        if password == password_true:
            loggedIn[username] = password
            return True
            # print("ture, log in sucessfully")
        else:
            return False
            # print("false, password wrong")


"""
登出成功返回true，登出失败返回false，同一个用户反复登出会失败(例如已经登出了，再次登出就
会失败)，本身就没有登录的用户登出也会失败。本身就没有注册的用户登出也会失败。
"""
def logout(username)->bool:
   global loggedIn
   # while True:
   if username in loggedIn:
       del loggedIn[username]
       return True
       # print("true, logout successfully")
   else:
       return False
       # print ("false, log out user not found")


"""
hint: 如何用这三个函数操作global的字典，registered = {} loggedIn = {}，logout信息
不需要单独存放在一个字典里也能实现。例如没有在login里面的要嘛就没登录要嘛就是没注册的。其他
的自己思考...
"""

"""
下面这个__main__只是给你调试输出上面三个函数功能实现是否正确准备的。我看你程序是否写的正确
是用另外的程序调用你的这三个函数进行测试，而不是看你下面这个__main__的输出。
"""
if __name__ == '__main__':
    reg_username = input("register username:", )
    reg_password = input("register password:", )
    register(reg_username, reg_password)
    log_username = input("log in username:", )
    log_password = input("log in password:", )
    login(log_username, log_password)
    logout_username = input("log out username:",)
    logout(logout_username)
    # print(registered)
    # print(loggedIn)
    # print("Your debug program entrance!")
