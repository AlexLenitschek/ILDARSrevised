#mohantys, converting  functions from mathematica to python
import os
import numpy as np
import sympy as sp
import time # for seeding the rng
import matplotlib.pyplot as plt
#import roomCreation as rc

# Fitting
#import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import math
from numpy.linalg import norm 
import random

# With given normalized vector v, u, w and the delta in m
def distanceWallDirection(u, v, w, delta):
	b = np.cross(np.cross(u, v), u)
	minor = np.dot(np.subtract(v, w), b) 
	p = 0
	altP = 0
	if (minor != 0): #minor != 0
		p = np.divide(np.multiply(np.dot(w, b), np.abs(delta)), minor)
	return p

# With given normalized vectors v, w, the vector n to wall and the delta in m
def distanceMapToNormal(n, v, w, delta):
	minor = np.dot(np.add(v, w), n)
	p = 0
	if (minor != 0): #minor != 0
		upper = np.dot(np.subtract(np.multiply(2, n), np.multiply(np.abs(delta), w)), n)
		p = np.divide(upper, minor)
	return p

# With given normalized vectors u, v, w
def distanceReflectionGeo(u, n, v, w):
	b = np.cross(np.cross(u, v), u)
	minor = np.add(np.multiply(np.dot(v, n), np.dot(w, b)),
				   np.multiply(np.dot(v, b), np.dot(w, n)))
	p = 0
	if (minor != 0): #minor != 0
		upper = np.multiply(2, np.multiply(np.dot(n, n), np.dot(w, b)))
		p = np.divide(upper, minor)
	return p

