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



def invert_3d_vector(x,y,z):
	return((x/(x**2 + y**2 + z**2)),(y/(x**2 + y**2 + z**2)),(z/(x**2 + y**2 + z**2)))


#print(invert_3d_vector(3,4,5))

#zW
def lengthOfVector(vector):
	res = 0
	for entry in vector:
		res = np.add(res, np.square(entry))
	res = np.sqrt(res)
	return res

# Input is a vector. Output is a vector with same orientation with length 1
#zw
def unifyVector(vector):
	if (lengthOfVector(vector) != 0):
		return np.divide(vector, lengthOfVector(vector))
	return vector

# Returns the angle betwwen two vectors a and b
#zW
def angleBetweenTwoVecor(v1, v2):
	uniV1 = unifyVector(v1)
	uniV2 = unifyVector(v2)
	return np.arccos(np.vdot(uniV1, uniV2))

def find_center_point(*points):

	ang = angleBetweenTwoVecor(points[[1]],points[[2]])
	co = np.cos(ang)/2
	centerpoint = {0, 0, 0} + ((preprocessing.normalize[(points[[1]] + points[[2]])/2 - {0, 0, 0}]))/co
	return(centerpoint,co)

def radianToDegree(radAngle):
	return np.multiply(radAngle, np.divide(180, np.pi))

# Returns a given angle in degree in radian
def degreeToRadian(degAngle):
	return np.multiply(degAngle, np.divide(np.pi, 180))


#issue in printing
#print(find_center_point(1,2,3))
def Calc_lat_long(*points):
	x = points[[1]]
	y = points[[2]]
	radius = math.sqrt(x**2 + y**2 + z**2)
	lat = np.ArcSin(z/radius)
	lon = np.ArcTan(x,y)
	return(int(latitude/Pi*180), int((longitude)/Pi*180), int(radius))


def cartesian_pts(*points):
	lat = points[[1]]
	lon = points[[2]]
	radius = points[[3]]
	return(int(radius*np.cos(radianToDegree(lat))*np.cos(radianToDegree(lon))), int(radius*np.cos(radianToDegree(lat))*np.sin(radianToDegree(lon))), int(radius*np.sin(radianToDegree(lat))))



def Circ_from_measurement(*measurement):
	v = measurement[1]
	w = measurement[2]
	delta = measurement[3]
	center = (delta/2) (w - v)/np.norm(w - v)^2
	radius = np.norm(center)
	normal = preprocessing.normalize(np.cross(v, w))
	return(center, radius, normal)



def segment_from_measurement(*measurement):
	Segments = {}
	v = measurement[[1]]
	w = measurement[[2]]
	delta = measurement[[3]]
	p1 = delta * (w - v)/np.normalize(w - v)**2
	p2 = delta/2*w
	return(p1, p2)


#in mathematica script, the below has an extra radius argument
#also, how to pass normal as an array?
# i guess in python *center and center can be passed alternatively ..
def invertCirc3D(center,normal):
  	mirrorPoint = center * 2
  	linePosition = InvertVector3D(mirrorPoint[[1]], mirrorPoint[[2]], mirrorPoint[[3]])
  	lineDirection = np.normalize(np.cross(linePosition, normal))
  	return(linePosition, lineDirection)
  



def mapping_2d(v):
	euc = math.sqrt(v[[1]]**2 + v[[2]]**2)
	return(v[[1]]/euc,v[[2]]/euc)

def mapping_3d(v):
	euc = math.sqrt(v[[1]]**2 + v[[2]]**2 + v[[3]]**2)
	return(v[[1]]/euc,v[[2]]/euc, v[[3]]/euc)


def GetVector(points):
	return(points[[1]], points[[2]] - points[[1]])


def GetVectors(lines):
	for i in range(1,len(lines)):
		vectors.append(GetVector(lines[[[i]]]))
	return(vectors)


#GNOMONIC PROJECTION :


#def GnomonicProjectionHemisphere(point,mappingpoint):
	#latmpoint = Calc_lat_long(mappingpoint)  yet to implement


# INTERSECTION HELPERS :



def OnSegment(p1,p2,p3):
	if(p2[[1]]) <= Max(p1[[1]], p3[[1]]) and p2[[1]] >= Min(p1[[1]], p3[[1]]) and p2[[2]] <= Max(p1[[2]], p3[[2]]) and p2[[2]] >= Min(p1[[2]], p3[[2]]):
		output = True
	else:
		output = False

	return(output)

# what does the orientation function do ? why did return have originally 1,2 for values > 0 ? 


def Orientation(p1,p2,p3):
	value = (p2[[2]] - p1[[2]])*(p3[[1]] - p2[[1]]) - (p2[[1]] - p1[[1]])*(p3[[2]] - p2[[2]])
	if(value==0):
		ret = 0
	if(value>0):
		ret = 1

	return(ret)


def DoIntersect(segment1,segment2):
	p1 = segment1[[1]]
	q1 = segment1[[2]]
	p2 = segment2[[1]] 
	q2 = segment2[[2]] 
	ret = False
	o1 = Orientation(p1, q1, p2)
	o2 = Orientation(p1, q1, q2)
	o3 = Orientation(p2, q2, p1)
	o4 = Orientation(p2, q2, q1)
	if(o1 != o2 and o3 != o4):
		ret = True;

	if(o1 == 0 and OnSegment(p1, p2, q1)):
		ret = True
	if(o2 == 0 and OnSegment(p1, q2, q1)):
		ret = True
	if(o3 == 0 and OnSegment(p2, p1, q2)):
		ret = True
	if(o4 == 0 and OnSegment(p2, q1, q2)):
		ret = True
	
	return(ret)


###############

#def IsPointOnFiniteLine({p1,p2},p3):
	#{minX, maxX} = np.sort({p1[1],p2[1]})
	#{minY, maxY} = np.sort({p1[2],p2[2]})
	#{minZ, maxZ} = np.sort({p1[3], p2[3]})


def NormalVectorFromTwoMeasurements(v1,w1,v2,w2):

	value = np.cross(np.cross(v1,w1),np.cross(v2,w2))
	if(value == {0,0,0}):
		value = 1
	return(value)

# for meeting : 09/11/2022		

def getCosC(point,mappingpoint):
	return(math.sin((radianToDegree(mappingpoint[1])))*math.sin((radianToDegree(point[1]))) + math.cos((radianToDegree(mappingpoint[1])))*math.cos(radianToDegree(point[1]))*math.cos(radianToDegree(point[2]-mappingpoint[2])))



def AdjustWallVectorLengthFromMeasurementsAndDirection(u,v,w,delta):
	r = p*v
	s = (p + delta+w)
	return(((r+s)/2)*u*u/Norm(u)**2)



def ComputeSenderDistanceWallDirection(v,w,delta,n):
	b = np.cross(np.cross(np.normalize(n),v),np.normalize(n))
	return(((delta*w)*b)/(v-w)*b)



def ComputeSenderDistanceMapToNormalVector(v,w,delta):
	return((2*n - delta*w)*n/(v+w)*n)

#ComputeSenderPositionClosestLinesExtended[vwnList_] :=  TO BE IMPLEMENTED


def AdjustWallVectorLengthFromMeasurementsAndDirection(u,v,w,delta):
	r = p*v
	s = (p + delta)*w
	rshalf = (r + s)/2
	rsdotu = (rshalf*u)
	t = rsdotu*u/norm(u)**2
	return(t)


def RandomOrthogonalVector(vector,angle):
	normalizedvector = vector/norm(vector)
	rearrangedvector = {-1*normalizedvector[3],normalizedvector[1],normalizedvector[2]}
	tangent = np.cross(normalizedvector,rearrangedvector)
	bitangent = np.cross(normalizedvector,tangent)
	randomvector = tangent*math.sin(angle) + bitangent*np.cos(angle)
	return(randomvector)

