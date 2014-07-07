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
	"""Read the data from the files"""
	def __init__(self,dirname,filename1,filename2):
		# First read from the file the coords per frame
		f1     = open(filename1,"r+")
		array1 = []
		for line in f1:
			alist    = line.split() 
			tmparray = []
			# read file: x,y,prediX,prediY,gtX,gtY
			for l in alist:
				tmparray.append(float(l))		
			array1.append(tmparray)	
		f1.close()
		f2     = open(filename2,"r+")
		array2 = []
		for line in f2:
			alist    = line.split() 
			tmparray = []
			# read file: x,y,prediX,prediY,gtX,gtY
			for l in alist:
				tmparray.append(float(l))		
			array2.append(tmparray)	
		f2.close()
		# Put the frame ids in X and 
                fig     = plt.figure()
                xCoord1 = []
                yCoord1 = []
		mean1   = 0

                xCoord2 = []
                yCoord2 = []
                mean2   = 0;		
		assert len(array1) == len(array2)
                for l in range(0,len(array1)):
			xCoord1.append(array1[l][0])
			yCoord1.append(array1[l][1])
			xCoord2.append(array2[l][0])
			yCoord2.append(array2[l][1])
			
			mean1 += array1[l][1]
			mean2 += array2[l][1]

		# Now plot both baseline and predicted
		mean1 /= len(array1)
		mean2 /= len(array2)

		yCoord1_mean = []
		yCoord2_mean = []
		minX = 1e+10
		maxX = 0
		minY = 1e+10
		maxY = 0
                for l in range(0,len(array1)):
			yCoord1[l] -= mean1
			yCoord2[l] -= mean2
			yCoord1_mean.append(mean1) 
			yCoord2_mean.append(mean2) 
			if(minX>xCoord1[l]):
				minX = xCoord1[l]
			if(minX>xCoord2[l]):
				minX = xCoord2[l]
			if(maxX<xCoord1[l]):
				maxX = xCoord1[l]
			if(maxX<xCoord2[l]):
				maxX = xCoord2[l]
			if(minY>yCoord1[l]):
				minY = yCoord1[l]
			if(minY>yCoord2[l]):
				minY = yCoord2[l]
			if(maxY<yCoord1[l]):
				maxY = yCoord1[l]
			if(maxY<yCoord2[l]):
				maxY = yCoord2[l]

		plt.title('End-Point-Errors per frame')
		plt.plot(xCoord2,yCoord2,'b-',linewidth=3)
		plt.plot(xCoord1,yCoord1,'r--',linewidth=3)
	
		plt.axis((minX,maxX,minY,maxY))	

		#plt.plot(xCoord1,yCoord1_mean,'r--')
		#plt.plot(xCoord2,yCoord2_mean,'b--')

		plt.legend(("Previous","Predicted"),loc='upper left')

		plt.xlabel('Frame Index')
		plt.ylabel('EPE score')
		plt.show()

            	pos  = string.rfind(dirname,PATH_SEP)
                path = dirname[0:pos]+PATH_SEP
                if(os.path.isdir(path)==False):
                        os.mkdir(path)
                figname = path+"unexpected.pdf"
                print figname 
                plt.savefig(figname)
#-------------------------------------------------------------------------------------------------------
print "command path/to/unexpected.txt command path/to/unexpected_prev.txt path/2/results"
print sys.argv
ms = MotionPlots(sys.argv[1],sys.argv[2],sys.argv[3]) 	
