import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt


class Sphere_Ellipsoid_surface:
	def __init__(someobj,xI,yI,zI,x1,y1,z1,x0,y0,z0,a,b,c):

		
		someobj.xI = xI
		someobj.yI = yI
		someobj.zI = zI
		someobj.x1 = x1
		someobj.y1 = y1
		someobj.z1 = z1
		someobj.x0 = x0
		someobj.y0 = y0
		someobj.z0 = z0
		someobj.a = a  # terms on the denominator of ellipsoid, for it's generic equation
		someobj.b = b   # terms on the denominator of ellipsoid, for it's generic equation
		someobj.c = c    # terms on the denominator of ellipsoid, for it's generic equation
		

	def sphere_tangent_normal(surface1): # sphere_calculations

	
	    #equation of curve
		print("Equation of tangent plane,  0 =",surface1.x0 - surface1.x1 ,"x +",+ surface1.y0 - surface1.y1,"y +", + surface1.z0 - surface1.z1 ,"z +", -x1*x0 -y0*y1 -z0*z1 + x1*x1 + y1*y1 + z1*z1)
		

		# so, basically a = x0 - x1 , b = y0 - y1 , c = z0 - z1,  d = -x0x1-y0y1 - z0z1 + x1*x1 + y1*y1 + z1*z1 

		a = surface1.x0 - surface1.x1
		b = surface1.y0 - surface1.y1
		c = surface1.z0 - surface1.z1
		d = -surface1.x0*surface1.x1 - surface1.y0*surface1.y1 - surface1.z0*surface1.z1 + surface1.x1*surface1.x1 + surface1.y1*surface1.y1 + surface1.z1*surface1.z1 


		t = - (a*surface1.xI + b*surface1.yI + c*surface1.zI + d)/(a*a + b*b + c*c)

		# t satisfying the mirror point, I' will be tnew :

		tnew = (-2* (a*surface1.xI + b*surface1.yI + c*surface1.zI + d))/(a*a + b*b + c*c) 

		xII = surface1.xI + tnew*a
		yII = surface1.yI + tnew*b
		zII = surface1.zI + tnew*c

		#incidence vector is :

		IP = [surface1.xI - surface1.x1, surface1.yI - surface1.y1, surface1.zI - surface1.z1]

		#Reflection vector is :

		IprimeP = [surface1.xII - surface1.x1, surface1.yII - surface1.y1, surface1.zII - surface1.z1]



	def ellipsoid_tangent_normal(surface1):

		print("Equation of tangent plane, 0 =" (surface1.x1 - surface1.x0)/surface1.a*surface1.a ,"x +",+ (surface1.y1 - surface1.y0)/surface1.b*surface1.b,"y +", + (surface1.z1 - surface1.z0)/surface1.c*surface1.c ,"z +", (-surface1.x1*surface1.x1)/(surface1.a*surface1.a) + (-surface1.y1*surface1.y1)/(surface1.b*surface1.b) + (-surface1.z1*surface1.z1)/(surface1.c*surface1.c) + (surface1.x0*surface1.x1)/(surface1.a*surface1.a) + (surface1.y0*surface1.y1)/(surface1.b*surface1.b) + (surface1.z0*surface1.z1)/(surface1.c*surface1.c))

		# so, basically the below calculates aplane, bplane, cplane and dplane, producing the generic equation of the tangent plane.

		aplane = (surface1.x1 - surface1.x0)/surface1.a*surface1.a 
		bplane = (surface1.y1 - surface1.y0)/surface1.b*surface1.b
		cplane = (surface1.z1 - surface1.z0)/surface1.c*surface1.c
		dplane = (-surface1.x1*surface1.x1)/(surface1.a*surface1.a) + (-surface1.y1*surface1.y1)/(surface1.b*surface1.b) + (-surface1.z1*surface1.z1)/(surface1.c*surface1.c) + (surface1.x0*surface1.x1)/(surface1.a*surface1.a) + (surface1.y0*surface1.y1)/(surface1.b*surface1.b) + (surface1.z0*surface1.z1)/(surface1.c*surface1.c)


		t = - (aplane*surface1.xI + bplane*surface1.yI + cplane*surface1.zI + dplane)/(aplane*aplane + bplane*bplane + cplane*cplane)

		# t satisfying the mirror point, I' will be tnew :

		tnew = 2*t

		xII = surface1.xI + tnew*aplane
		yII = surface1.yI + tnew*bplane
		zII = surface1.zI + tnew*cplane

		#incidence vector is :

		IP = [surface1.xI - surface1.x1, surface1.yI - surface1.y1, surface1.zI - surface1.z1]

		#Reflection vector is :

		IprimeP = [-(surface1.xII - surface1.x1), -(surface1.yII - surface1.y1), -(surface1.zII - surface1.z1)]
