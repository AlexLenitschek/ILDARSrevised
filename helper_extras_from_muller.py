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

def rotationMatrix(theta, vector):
	uVector = unifyVector(vector)
	# Defining Variables for shorter writing and readability
	ux, uy, uz = uVector[0], uVector[1], uVector[2]
	sin, cos = np.sin(theta), np.cos(theta)
	# Vectors of the matrix
	v1 = [cos+ux*ux*(1-cos)   , uy*ux*(1-cos)+uz*sin, uz*ux*(1-cos)-uy*sin]
	v2 = [ux*uy*(1-cos)-uz*sin, cos+uy*uy*(1-cos),    uz*uy*(1-cos)+ux*sin]
	v3 = [ux*uz*(1-cos)+uy*sin, uy*uz*(1-cos)-ux*sin, cos+uz*uz*(1-cos)]

	matrix = [v1, v2, v3]
	return matrix

 #Returns the length of given vector
def lengthOfVector(vector):
	res = 0
	for entry in vector:
		res = np.add(res, np.square(entry))
	res = np.sqrt(res)
	return res

# Input is a vector. Output is a vector with same orientation with length 1
def unifyVector(vector):
	if (lengthOfVector(vector) != 0):
		return np.divide(vector, lengthOfVector(vector))
	return vector

def isPointWithinRoom(point, room):
	if((room[2][0][0] > 0) and (room[2][0][1] > 0)):
		triangle1 = [room[0][0], [room[1][0][0]-room[2][0][0], 0, 0], room[2][0]]
		triangle2 = [[room[1][0][0]-room[2][0][0], 0, 0], room[1][0], room[2][0]]
		result = (isPointWithinTriangle(point, triangle1) or
				  isPointWithinTriangle(point, triangle2))
		return result
	else:
		return True

def isPointWithinTriangle(point, triangle):
	length = triangle[2][0] - triangle[1][0]
	width = triangle[3][1] - triangle[2][1]

	lengthRatio = np.divide(point[0], length)
	widthRatio = np.divide(point[1], width)
	if(lengthRatio + widthRatio > 1):
		return false
	return true

# Weight of each value is its distance to the wall
def weightedByDist(values, distance):
	if (np.sum(distance) == 0):
		return [0, 0, 0]
	result = [0, 0, 0]
	for i in range(len(values)):
		result += np.array(values[i])*np.abs(distance[i])
	result = result / np.sum(np.abs(distance))
	return result

# Weight of each value is its distance^3 to the wall
def avgPosition(values):
	if (len(values) == 0):
		return [0, 0, 0]
	result = [0, 0, 0]
	for i in range(len(values)):
		result += np.array(values[i])
	result = result / len(values)
	return result

# Root Mean Square Error
def rmse(predictionsMinusTargets):
    return np.sqrt(np.power(predictionsMinusTargets, 2).mean())



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

def find_center_point(points):

	ang = angleBetweenTwoVecor(points[[1]],points[[2]])
	co = np.cos(ang)/2
	centerpoint = {0, 0, 0} + ((preprocessing.normalize[(points[1] + points[2])/2 - {0, 0, 0}]))/co
	return(centerpoint,co)

def radianToDegree(radAngle):
	return np.multiply(radAngle, np.divide(180, np.pi))

# Returns a given angle in degree in radian
def degreeToRadian(degAngle):
	return np.multiply(degAngle, np.divide(np.pi, 180))


#issue in printing
#print(find_center_point(1,2,3))
# Explanation TODO; Function by Rico Gießler
def Calc_lat_long(points):
	y = points[[2]]
	radius = math.sqrt(x**2 + y**2 + z**2)
	lat = np.ArcSin(z/radius)
	lon = np.ArcTan(x,y)
	return(int(latitude/math.pi*180), int((longitude)/math.pi*180), int(radius))

# Explanation TODO; Function by Rico Gießler
def cartesian_pts(points):
	lat = points[1]
	lon = points[2]
	radius = points[3]
	return(int(radius*np.cos(radianToDegree(lat))*np.cos(radianToDegree(lon))), int(radius*np.cos(radianToDegree(lat))*np.sin(radianToDegree(lon))), int(radius*np.sin(radianToDegree(lat))))

# Compute a circle that intersects with the wall (as described in where's waldo talk), 
# defined by it's center radius and normal (vector perpendicular to the circle's surface).
# takes as input a measurement, i.e. a triple (v,w,Delta) where 
#   v is the direction of the direct signal
#   w is the direction of the reflected signal
#   Delta is the time difference between receiving the direct and the reflected signal.
def Circ_from_measurement(measurement):
	v = measurement[1]
	w = measurement[2]
	delta = measurement[3]
	center = (delta/2) (w - v)/np.norm(w - v)^2
	radius = np.norm(center)
	normal = preprocessing.normalize(np.cross(v, w))
	return(center, radius, normal)


# Compute a circular segment (instead of the completed circle) which intersects with the wall (as described in where's waldo talk).
# Since we only need the endpoints of the circular segment, only those will be computed, 
# even though for a complete geometric description of the circle segment, we would also need the center and radius.
# For computing radius and center, the previous function "Circ_from_measuremnt" could be used.
# takes as input a measurement, i.e. a triple (v,w,Delta) where 
#   v is the direction of the direct signal
#   w is the direction of the reflected signal
#   Delta is the time difference between receiving the direct and the reflected signal.
def segment_from_measurement(measurement):
	Segments = {}  # but this doesnt come of use here in the function definition or even the return value
	v = measurement[1]
	w = measurement[2]
	delta = measurement[3]
	p1 = delta * (w - v)/np.normalize(w - v)**2
	p2 = delta/2*w
	return(p1, p2)

# For a given circle (that goes through the orign by construction), compute it's inversion which is a line
# For a line l : p + x * d (where p is a point on l and d is a vector parallel to the line), we return
#   point p (called linePosition)
#   vector d (called lineDirection)
def invertCirc3D(center,normal):
  	mirrorPoint = center * 2
  	linePosition = InvertVector3D(mirrorPoint[1], mirrorPoint[2], mirrorPoint[3])
  	lineDirection = np.normalize(np.cross(linePosition, normal)) # This will not work, as there is no np.normalize
  	return(linePosition, lineDirection)

# Explanation TODO; Function by Rico Gießler
def mapping_2d(v):
	euc = math.sqrt(v[1]**2 + v[2]**2)
	return(v[1]/euc,v[2]/euc)

# Explanation TODO; Function by Rico Gießler
def mapping_3d(v):
	euc = math.sqrt(v[1]**2 + v[2]**2 + v[3]**2)
	return(v[1]/euc,v[2]/euc, v[3]/euc)

# Explanation TODO; Function by Rico Gießler
def GetVector(points):
	return(points[1], points[2] - points[1])

# Explanation TODO; Function by Rico Gießler
def GetVectors(lines):
	for i in range(1,len(lines)):
		vectors.append(GetVector(lines[[i]]))
	return(vectors)


#GNOMONIC PROJECTION :


#def GnomonicProjectionHemisphere(point,mappingpoint):
	#latmpoint = Calc_lat_long(mappingpoint)  yet to implement


# INTERSECTION HELPERS :



# Explanation TODO; Function by Rico Gießler
def OnSegment(p1,p2,p3):
	if(p2[1]) <= Max(p1[1], p3[1]) and p2[1] >= Min(p1[1], p3[1]) and p2[2] <= Max(p1[2], p3[2]) and p2[2] >= Min(p1[2], p3[2]):
		output = True
	else:
		output = False

	return(output)

# what does the orientation function do ? why did return have originally 1,2 for values > 0 ? 
# Explanation TODO; Function by Rico Gießler
def Orientation(p1,p2,p3):
	value = (p2[2] - p1[2])*(p3[1] - p2[1]) - (p2[1] - p1[1])*(p3[2] - p2[2])
	if(value==0):
		ret = 0
	if(value>0):
		ret = 1

	return(ret)

# Explanation TODO; Function by Rico Gießler
def DoIntersect(segment1,segment2):
	p1 = segment1[1]
	q1 = segment1[2]
	p2 = segment2[1] 
	q2 = segment2[2] 
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


# Explanation TODO; Function by Rico Gießler
def NormalVectorFromTwoMeasurements(v1,w1,v2,w2):
	value = np.cross(np.cross(v1,w1),np.cross(v2,w2))
	if(value == {0,0,0}):
		value = 1
	return(value)

# for meeting : 09/11/2022		

# Explanation TODO; Function by Rico Gießler
def getCosC(point,mappingpoint):
	return(math.sin((radianToDegree(mappingpoint[1])))*math.sin((radianToDegree(point[1]))) + math.cos((radianToDegree(mappingpoint[1])))*math.cos(radianToDegree(point[1]))*math.cos(radianToDegree(point[2]-mappingpoint[2])))

def AdjustWallVectorLengthFromMeasurementsAndDirection(u,v,w,delta):
	r = p*v
	s = (p + delta+w)
	return(((r+s)/2)*u*u/np.norm(u)**2)

# Given a measurement (v,w,Delta) and the normal vector n of a wall (assuming that w was reflect from the wall for which n is the normal vector),
# we can compute the distance of the sender using the "Wall Direction" formula as described in Proto-ILDARS Talk/Slides.
def ComputeSenderDistanceWallDirection(v,w,delta,n):
	b = np.cross(np.cross(np.normalize(n),v),np.normalize(n))
	return(((delta*w)*b)/(v-w)*b)

# MM: Something is not right here, I assume the parameter n is missing
# Other than that, this function is an implementation of the "Map to normal wall vector" formula described in Proto-ILDARS Talk/Slides.
def ComputeSenderDistanceMapToNormalVector(v,w,n,delta):
	return((2*n - delta*w)*n/(v+w)*n)

#ComputeSenderPositionClosestLinesExtended[vwnList_] :=  TO BE IMPLEMENTED

# MM: I think this was implemented by Tobias Grugel and is used for wall normal vector computation. I will have to confirm that though.
# TODO: Explanation
def AdjustWallVectorLengthFromMeasurementsAndDirection(u,v,w,delta):
	r = p*v
	s = (p + delta)*w
	rshalf = (r + s)/2
	rsdotu = (rshalf*u)
	t = rsdotu*u/norm(u)**2
	return(t)

# MM: Not quite sure who implemented this, I think the function is used for Error simulation
# Explanation TODO
def RandomOrthogonalVector(vector,angle):
	normalizedvector = vector/norm(vector)
	rearrangedvector = {-1*normalizedvector[3],normalizedvector[1],normalizedvector[2]}
	tangent = np.cross(normalizedvector,rearrangedvector)
	bitangent = np.cross(normalizedvector,tangent)
	randomvector = tangent*math.sin(angle) + bitangent*np.cos(angle)
	return(randomvector)



def MapToUnitCircleWithWall(L):
	localL = []
	for i  in range(len(L)):
		localL.append(Mapping(L[i,1]), Mapping(L[i,2]), Mapping(L[i,3]),Mapping(L[i,4]))
		return(localL)

# split original mathematica code into 2 parts

def Get12Hemispheres_zaxis():
	x = random.random(math.pi)
	vec = Normalize[1, 0, 0]
  	vec2 = Normalize[0, 1, 0]
  	vec3 = Normalize[0, 0, 1]
  	vec4 = Normalize[1, 1, 1]
  	vec5 = Normalize[1, 1, -1]
  	vec6 = Normalize[1, -1, -1]
  	vec = [vec[1]*math.cos(x) - vec[2]*math.sin(x), 
    vec[1]*math.sin(x) + vec[2]*math.cos(x), vec[3]]
  	vec2 = [vec2[1]*math.cos(x) - vec2[2]*math.sin(x), 
    vec2[1]*math.sin(x) + vec2[2]*math.cos(x), vec2[3]]
    vec3 = [vec3[1]*math.cos(x) - vec3[2]*math.sin(x), 
    vec3[1]*math.sin(x) + vec3[2]*math.cos(x), vec3[3]]
  	vec4 = [vec4[1]*math.cos(x) - vec4[2]*math.sin(x), 
    vec4[1]*math.sin(x) + vec4[2]*math.cos(x), vec4[3]]
    vec5 = [vec5[1]*math.cos(x) - vec5[2]*math.sin(x), 
    vec5[1]*math.sin(x) + vec5[2]*math.cos(x), vec5[3]]
    vec6 = [vec6[1]*math.cos(x) - vec6[2]*math.sin(x), 
    vec6[1]*math.sin(x) + vec6[2]*math.cos(x), vec6[3]]

    return(vec, -vec, vec2, -vec2, vec3, -vec3, vec4, -vec4, vec5, -vec5, vec6, -vec6)

def Get12Hemispheres_yaxis():
	x = random.random(math.pi)
	vec = Normalize[1, 0, 0]
  	vec2 = Normalize[0, 1, 0]
  	vec3 = Normalize[0, 0, 1]
  	vec4 = Normalize[1, 1, 1]
  	vec5 = Normalize[1, 1, -1]
  	vec6 = Normalize[1, -1, -1]
 
  	vec = [vec[1]*math.cos(x) + vec[3]*math.sin(x), 
    vec[2], -vec[1]*math.sin(x) + vec[3]*math.cos(x)]
  	vec2 = [vec2[1]*math.cos(x) + vec2[3]*math.sin(x), 
    vec2[2], -vec2[1]*math.sin(x) + vec2[3]*math.cos(x)]
    vec3 = [vec3[1]*math.cos(x) + vec3[3]*math.sin(x), 
    vec3[2], -vec3[1]*math.sin(x) + vec3[3]*math.cos(x)]
    vec4 = [vec4[1]*math.cos(x) + vec4[3]*math.sin(x), 
    vec4[2], -vec4[1]*math.sin(x) + vec4[3]*math.cos(x)]
    vec5 = [vec5[1]*math.cos(x) + vec5[3]*math.sin(x), 
    vec5[2], -vec5[1]*math.sin(x) + vec5[3]*math.cos(x)]
    vec6 = [vec6[1]*math.cos(x) + vec6[3]*math.sin(x), 
    vec6[2], -vec6[1]*math.sin(x) + vec6[3]*math.cos(x)]

    return(vec, -vec, vec2, -vec2, vec3, -vec3, vec4, -vec4, vec5, -vec5, vec6, -vec6)


def ComputeMirrorPoint(triangle,point):
	TriangleNormal = np.normalize[np.cross[triangle[2] - triangle[1], triangle[3] - triangle[1]]]
	mirrorPoint = point + (((triangle[1] - point) . triangleNormal)*triangleNormal*2)
  	return(mirrorPoint)