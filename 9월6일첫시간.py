# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

list = ['adb', 34, 3.234]
#type 'list' can contain string, integer, real number etc.
#index starts with 0
#리스트는 메모리 크기가 크게 할당된다.
#slicing슬라이씽
tuple = ('abc', 456, 70.2)
#튜플은 사이즈 지정되어있다.딱 이만큼 메모리 할당.

#리스트와 튜플은 인덱스 사용하여 각각에 접근할 수 있지만,
#딕셔너리는 키와 발류로 구성되어 있어서 키로 접근.
dict = {}
dict['one'] = "this is one"
dict[2] = "this is two"

print(dict)

for x in range(5):
    print(x)
    
print('---------')

primes = [2,3,5,7]
for prime in primes:
    print(prime)
    
    
A = [x**2 for x in range(10)]# **2 이것은 제곱.
print(A)

A = [x**2 for x in range(10) if x%2 == 0]
print(A)

A = [x**2 if x%2==0 else x+1 for x in range(10)]
print(A)

A=[8,7,5,13,75,65,11]
print(A[:2])
print(A[-2:])

