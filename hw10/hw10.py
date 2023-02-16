import cv2 
import math
import cmath
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import measure
import random

img = cv2.imread("lena.bmp")
height=img.shape[0]
width=img.shape[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.bmp", gray) 
#二值化
binary=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(height):
    for j in range(width):
        if binary[i,j]<128:
            binary[i,j]=0
        elif binary[i,j]>=128:
            binary[i,j]=255
cv2.imwrite("binary.bmp", binary) 
print(binary[0][0])

def laplacian(img):
    threshold=15
    filter1=[[0,1,0],
             [1,-4,1],
             [0,1,0]]
    filter2=[[1/3,1/3,1/3],
             [1/3,-8/3,1/3],
             [1/3,1/3,1/3]]
    mask=np.zeros((512,512),dtype=int)
    laplacian1=np.zeros((512,512),dtype=int)
    laplacian2=np.zeros((512,512),dtype=int)
    height=img.shape[0]
    width=img.shape[1]
    border_img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp=0
            for m in range(3):
                for n in range(3):
                    tmp=tmp+filter1[m][n]*border_img[i+m][j+n]
            mask[i][j]=tmp
            if(mask[i][j]>=threshold):
                mask[i][j]=1
            elif(mask[i][j]<=(-threshold)):
                mask[i][j]=-1
            elif(threshold>=mask[i][j]>=(-threshold)):
                mask[i][j]=0
    border_mask=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            if(border_mask[i+1][j+1]>=1):
                laplacian1[i][j]=255
                for m in range(3):
                    for n in range(3):
                        if(border_mask[i+m][j+n]<=-1):
                            laplacian1[i][j]=0
            else:
                laplacian1[i][j]=255
    laplacian1_img=Image.fromarray(laplacian1)
    laplacian1_img=laplacian1_img.convert("L")  
    laplacian1_img.save("laplacian1_img.bmp")   
    
    for i in range(height):
        for j in range(width):
            tmp=0
            for m in range(3):
                for n in range(3):
                    tmp=tmp+filter2[m][n]*border_img[i+m][j+n]
            mask[i][j]=tmp
            if(mask[i][j]>=threshold):
                mask[i][j]=1
            elif(mask[i][j]<=(-threshold)):
                mask[i][j]=-1
            elif(threshold>=mask[i][j]>=(-threshold)):
                mask[i][j]=0
    border_mask=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            if(border_mask[i+1][j+1]>=1):
                laplacian2[i][j]=255
                for m in range(3):
                    for n in range(3):
                        if(border_mask[i+m][j+n]<=-1):
                                laplacian2[i][j]=0         
            else:
                laplacian2[i][j]=255
    laplacian2_img=Image.fromarray(laplacian2)
    laplacian2_img=laplacian2_img.convert("L")  
    laplacian2_img.save("laplacian2_img.bmp")   
    
laplacian(gray)    


def Minimum_variance_Laplacian(img): 
    threshold=20
    filter1=[[2/3,-1/3,2/3],
             [-1/3,-4/3,-1/3],
             [2/3,-1/3,2/3]]
    mask=np.zeros((512,512),dtype=int)
    mv_laplacian=np.zeros((512,512),dtype=int)
    height=img.shape[0]
    width=img.shape[1]
    border_img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp=0
            for m in range(3):
                for n in range(3):
                    tmp=tmp+filter1[m][n]*border_img[i+m][j+n]
            mask[i][j]=tmp
            if(mask[i][j]>=threshold):
                mask[i][j]=1
            elif(mask[i][j]<=(-threshold)):
                mask[i][j]=-1
            elif(threshold>=mask[i][j]>=(-threshold)):
                mask[i][j]=0
    border_mask=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            if(border_mask[i+1][j+1]>=1):
                mv_laplacian[i][j]=255
                for m in range(3):
                    for n in range(3):
                        if(border_mask[i+m][j+n]<=-1):
                            mv_laplacian[i][j]=0
            else:
                mv_laplacian[i][j]=255
    mv_laplacian_img=Image.fromarray(mv_laplacian)
    mv_laplacian_img=mv_laplacian_img.convert("L")  
    mv_laplacian_img.save("Minimum_variance_Laplacian_img.bmp") 
    
Minimum_variance_Laplacian(gray)    
    
def Laplacian_of_Gaussian(img): 
    threshold=3000
    filter1=[[0,0,0,-1,-1,-2,-1,-1,0,0,0],
             [0,0,-2,-4,-8,-9,-8,-4,-2,0,0],
             [0,-2,-7,-15,-22,-23,-22,-15,-7,-2,0],
             [-1,-4,-15,-24,-14,-1,-14,-24,-15,-4,-1],
             [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1],
             [-2,-9,-23,-1,103,178,103,-1,-23,-9,-2],
             [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1],
             [-1,-4,-15,-24,-14,-1,-14,-24,-15,-4,-1],
             [0,-2,-7,-15,-22,-23,-22,-15,-7,-2,0],
             [0,0,-2,-4,-8,-9,-8,-4,-2,0,0],
             [0,0,0,-1,-1,-2,-1,-1,0,0,0]]
    mask=np.zeros((512,512),dtype=int)
    LaplacianofGaussian=np.zeros((512,512),dtype=int)
    height=img.shape[0]
    width=img.shape[1]
    border_img=cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp=0
            for m in range(11):
                for n in range(11):
                    tmp=tmp+filter1[m][n]*border_img[i+m][j+n]
            mask[i][j]=tmp
            if(mask[i][j]>=threshold):
                mask[i][j]=1
            elif(mask[i][j]<=(-threshold)):
                mask[i][j]=-1
            elif(threshold>=mask[i][j]>=(-threshold)):
                mask[i][j]=0
    border_mask=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            if(border_mask[i+1][j+1]>=1):
                LaplacianofGaussian[i][j]=255
                for m in range(3):
                    for n in range(3):
                        if(border_mask[i+m][j+n]<=-1):
                            LaplacianofGaussian[i][j]=0
            else:
                LaplacianofGaussian[i][j]=255
    LaplacianofGaussian_img=Image.fromarray(LaplacianofGaussian)
    LaplacianofGaussian_img=LaplacianofGaussian_img.convert("L")  
    LaplacianofGaussian_img.save("Laplacian_of_Gaussian_img.bmp") 
    
Laplacian_of_Gaussian(gray)    
    
    
def Difference_of_Gaussian(img): 
    threshold=1
    filter1=[[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]
    mask=np.zeros((512,512),dtype=int)
    DifferenceofGaussian=np.zeros((512,512),dtype=int)
    height=img.shape[0]
    width=img.shape[1]
    border_img=cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp=0
            for m in range(11):
                for n in range(11):
                    tmp=tmp+filter1[m][n]*border_img[i+m][j+n]
            mask[i][j]=tmp
            if(mask[i][j]>=threshold):
                mask[i][j]=1
            elif(mask[i][j]<=(-threshold)):
                mask[i][j]=-1
            elif(threshold>=mask[i][j]>=(-threshold)):
                mask[i][j]=0
    border_mask=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            if(border_mask[i+1][j+1]>=1):
                DifferenceofGaussian[i][j]=255
                for m in range(3):
                    for n in range(3):
                        if(border_mask[i+m][j+n]<=-1):
                            DifferenceofGaussian[i][j]=0
            else:
                DifferenceofGaussian[i][j]=255
    DifferenceofGaussian_img=Image.fromarray(DifferenceofGaussian)
    DifferenceofGaussian_img=DifferenceofGaussian_img.convert("L")  
    DifferenceofGaussian_img.save("Difference_of_Gaussian_img.bmp") 
    
Difference_of_Gaussian(gray)        