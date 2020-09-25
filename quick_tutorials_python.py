# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

# string assignment
data = 'hello world'
print(data[0])
print(len(data))
print(data)

#number assignment
value = 123.1
print(value)
value = 10
print(value)

#boolean assignment
a = True
b = False
print(a, b)

#multiple assignment
d, e, f = 1, 2, 3
print(d, e, f)

#none assignment
g = None
print(g)

#flow control statements
#if-then-else conditional statement
value = 205
if value == 199:
    print ('That is fast')
elif value > 200:
    print ('That is too fast')
else:
    print ('That is safe')

#for loop
for i in range(10):
    print (i)

#while loop
i = 0
while i<10:
    print (i)
    i += 1
    
#data structures
#tuples
a = (1, 2, 3)
print(a)    

#list
mylist = [1, 2, 3]
print("zeroth value: %d" % mylist[0]) 
mylist.append(4)
print("List length: %d" % len(mylist)) 
for value in mylist:
    print (value)

#dictionary
mydict = {'a': 1, 'b': 2, 'c': 3}
print("A value: %d" % mydict['a'])
mydict['a'] = 11
print("A value: %d" % mydict['a'])
print("keys: %s" % mydict.keys())
print("Values: %s" % mydict.values())
for key in mydict.keys():
    print (mydict[key])

#function
def mysum(x, y):
    return x + y

#call the function
result = mysum(5, 3)
print(result)







    















