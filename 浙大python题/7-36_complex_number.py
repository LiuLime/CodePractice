
class Complex:
    def __init__(self):
        pass

    def output(self, complex_input):
        self.complex_input = complex_input
        rel = self.complex_input.real
        imag = self.complex_input.imag
        c = complex(rel, imag)
        c_real = round(c.real, 1)
        c_imag = round(c.imag, 1)
        result = format(complex(c), '.1f').replace('j', 'i')
        if c_real == 0 and c_imag != 0:
            result = format(imag, '.1f')+'i'
        elif c_real != 0 and c_imag ==0:
            result = format(rel, '.1f')
        elif c_real == 0 and c_imag ==0:
            result = format(0.0, '.1f')
        else:
            pass
        return result


a = input('').split()
#不能在这里就round，会影响计算
a1 = float(a[0])
a2 = float(a[2])
b1 = float(a[1])
b2 = float(a[3])

C1 = complex(a1, b1)
C2 = complex(a2, b2)
C1_round = format(C1, '.1f').replace("j", 'i')
C2_round = format(C2, '.1f').replace('j', 'i')
C1_round = f'({C1_round})'
C2_round = f'({C2_round})'

result1 = C1 + C2
result2 = C1 - C2
result3 = C1 * C2
result4 = C1 / C2

com = Complex()
result1 = com.output(result1)
result2 = com.output(result2)
result3 = com.output(result3)
result4 = com.output(result4)

print(C1_round, '+', C2_round, '=', result1)
print(C1_round, '-', C2_round, '=', result2)
print(C1_round, '*', C2_round, '=', result3)
print(C1_round, '/', C2_round, '=', result4, end='')
#问题： 表示的时候四舍五入了，但是计算的时候实际数字没变