import os,sys,math,string
import matplotlib.backends.backend_tkagg
from matplotlib import *
import matplotlib.pyplot as plt
from matplotlib import collections, transforms
from matplotlib.colors import colorConverter
import numpy as np
import scipy.misc
PATH_SEP = "/"
class MotionPlots:
	dirname  = ""
	dirims   = ""
	filename = ""
	imfile   = ""
	array    = []
	ext      = ""
	"""Read the data from the files"""
	def __init__(self,inname,imspath,ext,option):
		self.dirname = inname
		self.dirims  = imspath
		self.ext     = ext
		if(self.dirname[len(self.dirname)-1]==PATH_SEP):
			self.dirname = self.dirname[0:len(self.dirname)-1]	
		allfiles     = os.listdir(self.dirname)	
		for filename in allfiles:
			self.filename = self.dirname+PATH_SEP+filename
			self.imfile   = self.dirims+PATH_SEP+filename[2:len(filename)-3]+self.ext
			f             = open(self.filename,"r+")
			self.array    = []
			for line in f:
				alist    = line.split() 
				tmparray = []
				# read file: x,y,prediX,prediY,gtX,gtY
				for l in alist:
					tmparray.append(float(l))		
				self.array.append(tmparray)	
			f.close()
			# See which plot we do
			if(option=="scatter"):
				self.scatter(filename)
			elif(option=="heatmap"):
				self.heatmap(filename)
			self.array = []
	"""Scatter the prediction magnitudes versus error in predi color."""
	def scatter(self,justname):
		fig    = plt.figure()
		magniG = []
		magniB = []
		magniR = []
		magniY = []
		errorG = []
		errorB = []
		errorR = []
		errorY = []
		# 0 - x coord, 1 - y coord, 2 - predicted x-flow, 3 - predicted y-flow, 4 - estimated x-flow, 5 - estimated y-flow
		for l in range(0,len(self.array)):
			angle = math.atan2(self.array[l][3],self.array[l][2])+math.pi
			if(angle<=math.pi/2.0):	
				magniG.append(math.sqrt(self.array[l][4]*self.array[l][4]+self.array[l][5]*self.array[l][5]))
				errorG.append(math.sqrt((self.array[l][2]-self.array[l][4])*(self.array[l][2]-self.array[l][4])+
                                	(self.array[l][3]-self.array[l][5])*(self.array[l][3]-self.array[l][5])))
			elif(angle<=math.pi):
				magniR.append(math.sqrt(self.array[l][4]*self.array[l][4]+self.array[l][5]*self.array[l][5]))
				errorR.append(math.sqrt((self.array[l][2]-self.array[l][4])*(self.array[l][2]-self.array[l][4])+
                                	(self.array[l][3]-self.array[l][5])*(self.array[l][3]-self.array[l][5])))
			elif(angle<=3.0*math.pi/2.0):
				magniB.append(math.sqrt(self.array[l][4]*self.array[l][4]+self.array[l][5]*self.array[l][5]))
				errorB.append(math.sqrt((self.array[l][2]-self.array[l][4])*(self.array[l][2]-self.array[l][4])+
                                	(self.array[l][3]-self.array[l][5])*(self.array[l][3]-self.array[l][5])))
			else:
				magniY.append(math.sqrt(self.array[l][4]*self.array[l][4]+self.array[l][5]*self.array[l][5]))
				errorY.append(math.sqrt((self.array[l][2]-self.array[l][4])*(self.array[l][2]-self.array[l][4])+
                                	(self.array[l][3]-self.array[l][5])*(self.array[l][3]-self.array[l][5])))
		plt.title('Prediction Scatter Plot')
		plt.ylabel('EPE (in pixels)')
		plt.xlabel('Estimated magnitude')
		plt.axis([0,8,0,8])
		g    = plt.scatter(magniG,errorG,s=30,c="green",marker='.',cmap=None,norm=None,vmin=None,vmax=None,alpha=1,edgecolors="green")
		r    = plt.scatter(magniR,errorR,s=30,c="red",marker='.',cmap=None,norm=None,vmin=None,vmax=None,alpha=1,edgecolors="red")
		b    = plt.scatter(magniB,errorB,s=30,c="blue",marker='.',cmap=None,norm=None,vmin=None,vmax=None,alpha=1,edgecolors="blue")
		y    = plt.scatter(magniY,errorY,s=30,c="yellow",marker='.',cmap=None,norm=None,vmin=None,vmax=None,alpha=1,edgecolors="yellow")
		plt.legend([g,r,b,y],["Predicted Quadrant 0","Predicted Quadrant 2","Predicted Quadrant 3","Predicted Quadrant 4"],loc=2)
		pos  = string.rfind(self.dirname,PATH_SEP)
		path = self.dirname[0:pos]+PATH_SEP+"scatters"+PATH_SEP
		if(os.path.isdir(path)==False):
			os.mkdir(path)
		figname = path+justname[0:len(justname)-4]+".pdf"
                print figname 
		plt.savefig(figname)
	"""A heatmap with error-levels per pixels."""
	def heatmap(self,justname):
                fig      = plt.figure()
                xCoord   = []
                yCoord   = []
		image    = scipy.misc.imread(self.imfile)
		size     = 300.0/max(image.shape[0],image.shape[1])
		image    = scipy.misc.imresize(image,size)
		error    = numpy.zeros((image.shape[0],image.shape[1]))
		sumError = 0;
                for l in range(0,len(self.array)):
			xCoord.append(self.array[l][1])
			yCoord.append(self.array[l][0])
			tmperror = 0
			if(self.array[l][0]>50 and self.array[l][0]<image.shape[1]-50):
              			tmperror = math.sqrt((self.array[l][4]-self.array[l][2])*(self.array[l][4]-self.array[l][2])+
                               		(self.array[l][5]-self.array[l][3])*(self.array[l][5]-self.array[l][3]))          
			error[int(self.array[l][1])][int(self.array[l][0])] = tmperror               
			#if(sumError<tmperror):
			sumError += tmperror
		error -= sumError/float(len(self.array))
		plt.title('Heatmap for End-Point-Errors')
                plt.imshow(image,interpolation='nearest')
		plt.hold(True)
                plt.imshow(error,cmap=cm.jet,interpolation='nearest',alpha=0.6)
		#plt.show()

            	pos  = string.rfind(self.dirname,PATH_SEP)
                path = self.dirname[0:pos]+PATH_SEP+"heatmaps"+PATH_SEP
                if(os.path.isdir(path)==False):
                        os.mkdir(path)
                figname = path+justname[0:len(justname)-4]+".png"
                print figname 
                plt.savefig(figname)
#-------------------------------------------------------------------------------------------------------
print "command path/to/coordinate_file.txt path/2/images/ extension [scatter|heatmap]"
ms = MotionPlots(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]) 	
