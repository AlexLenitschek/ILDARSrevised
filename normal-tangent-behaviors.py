#normal-tangent behaviors
import os
import numpy as np
import sympy as sp
import time # for seeding the rng
import matplotlib.pyplot as plt


# Fitting

'''
p = input("Enter the distance between the source and target:")
print("wall reflection vector is: " + p)
v = input("Enter wall vector:")
print("Wall vector is: " + v)
w = input("Enter the reflection from wall vector:")
print("Wall vector is: " + w)
'''

p = 4
v = 1
w = 1
delta = 2


r = np.multiply((p + delta),w)
s = np.multiply(p,v)


n = np.multiply(delta,(p + delta/2))/(abs(np.multiply((p+delta),w) - np.multiply(p,v)))

print('magnitude of normal', +n)

t = ((r + s)/2) - n

print('magnitude of tangent', +t)

