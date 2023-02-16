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


def roberts(img):
    robert_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    r =np.zeros((2,2),dtype=int)
    r=[[-1,-1],
       [1,1]]
    for i in range(height-1):
        for j in range(width-1):            
            tmp1=(img[i+1][j+1]*r[1][1]+img[i][j]*r[0][0])*(img[i+1][j+1]*r[1][1]+img[i][j]*r[0][0])
            tmp2=(img[i+1][j]*r[1][0]+img[i][j+1]*r[0][1])*(img[i+1][j]*r[1][0]+img[i][j+1]*r[0][1])
            robert_img[i][j]=math.sqrt(tmp1+tmp2)
            if robert_img[i][j]>=12:
                robert_img[i][j]=0
            else:
                robert_img[i][j]=255
    return robert_img

roberts_img=roberts(gray) 
new_roberts_img=Image.fromarray(roberts_img)
new_roberts_img=new_roberts_img.convert("L")  
new_roberts_img.save("new_roberts_img.bmp")    

def prewitt(img):
    prewitt_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    p1=[[-1,-1,-1],
        [0,0,0],
        [1,1,1]]
    p2=[[-1,0,1],
        [-1,0,1],
        [-1,0,1]]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):            
            tmp1=(img[i][j]*p1[0][0]+img[i][j+1]*p1[0][1]+img[i][j+2]*p1[0][2]+
                  img[i+1][j]*p1[1][0]+img[i+1][j+1]*p1[1][1]+img[i+1][j+2]*p1[1][2]+
                  img[i+2][j]*p1[2][0]+img[i+2][j+1]*p1[2][1]+img[i+2][j+2]*p1[2][2])**2
            tmp2=(img[i][j]*p2[0][0]+img[i][j+1]*p2[0][1]+img[i][j+2]*p2[0][2]+
                  img[i+1][j]*p2[1][0]+img[i+1][j+1]*p2[1][1]+img[i+1][j+2]*p2[1][2]+
                  img[i+2][j]*p2[2][0]+img[i+2][j+1]*p2[2][1]+img[i+2][j+2]*p2[2][2])**2
            prewitt_img[i][j]=math.sqrt(tmp1+tmp2)
            if prewitt_img[i][j]>=24:
                prewitt_img[i][j]=0
            else:
                prewitt_img[i][j]=255
    return prewitt_img

prewitt_img=prewitt(gray) 
new_prewitt_img=Image.fromarray(prewitt_img)
new_prewitt_img=new_prewitt_img.convert("L")  
new_prewitt_img.save("new_prewitt_img.bmp")    

def sobel(img):
    sobel_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    p1=[[-1,-2,-1],
        [0,0,0],
        [1,2,1]]
    p2=[[-1,0,1],
        [-2,0,2],
        [-1,0,1]]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):            
            tmp1=(img[i][j]*p1[0][0]+img[i][j+1]*p1[0][1]+img[i][j+2]*p1[0][2]+
                  img[i+1][j]*p1[1][0]+img[i+1][j+1]*p1[1][1]+img[i+1][j+2]*p1[1][2]+
                  img[i+2][j]*p1[2][0]+img[i+2][j+1]*p1[2][1]+img[i+2][j+2]*p1[2][2])**2
            tmp2=(img[i][j]*p2[0][0]+img[i][j+1]*p2[0][1]+img[i][j+2]*p2[0][2]+
                  img[i+1][j]*p2[1][0]+img[i+1][j+1]*p2[1][1]+img[i+1][j+2]*p2[1][2]+
                  img[i+2][j]*p2[2][0]+img[i+2][j+1]*p2[2][1]+img[i+2][j+2]*p2[2][2])**2
            sobel_img[i][j]=math.sqrt(tmp1+tmp2)
            if sobel_img[i][j]>=38:
                sobel_img[i][j]=0
            else:
                sobel_img[i][j]=255
    return sobel_img

sobel_img=sobel(gray) 
new_sobel_img=Image.fromarray(sobel_img)
new_sobel_img=new_sobel_img.convert("L")  
new_sobel_img.save("new_sobel_img.bmp")    

def freiandchen(img):
    freiandchen_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    tmp=math.sqrt(2)
    p1=[[-1,-tmp,-1],
        [0,0,0],
        [1,tmp,1]]
    p2=[[-1,0,1],
        [-tmp,0,tmp],
        [-1,0,1]]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):            
            tmp1=(img[i][j]*p1[0][0]+img[i][j+1]*p1[0][1]+img[i][j+2]*p1[0][2]+
                  img[i+1][j]*p1[1][0]+img[i+1][j+1]*p1[1][1]+img[i+1][j+2]*p1[1][2]+
                  img[i+2][j]*p1[2][0]+img[i+2][j+1]*p1[2][1]+img[i+2][j+2]*p1[2][2])**2
            tmp2=(img[i][j]*p2[0][0]+img[i][j+1]*p2[0][1]+img[i][j+2]*p2[0][2]+
                  img[i+1][j]*p2[1][0]+img[i+1][j+1]*p2[1][1]+img[i+1][j+2]*p2[1][2]+
                  img[i+2][j]*p2[2][0]+img[i+2][j+1]*p2[2][1]+img[i+2][j+2]*p2[2][2])**2
            freiandchen_img[i][j]=math.sqrt(tmp1+tmp2)
            if freiandchen_img[i][j]>=30:
                freiandchen_img[i][j]=0
            else:
                freiandchen_img[i][j]=255
    return freiandchen_img

freiandchen_img=freiandchen(gray) 
new_freiandchen_img=Image.fromarray(freiandchen_img)
new_freiandchen_img=new_freiandchen_img.convert("L")  
new_freiandchen_img.save("new_freiandchen_img.bmp")  

def kirsch(img):
    kirsch_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    k0=[[-3,-3,5],
        [-3,0,5],
        [-3,-3,5]]
    k1=[[-3,5,5],
        [-3,0,5],
        [-3,-3,-3]]
    k2=[[5,5,5],
        [-3,0,-3],
        [-3,-3,-3]]
    k3=[[5,5,-3],
        [5,0,-3],
        [-3,-3,-3]]
    k4=[[5,-3,-3],
        [5,0,-3],
        [5,-3,-3]]
    k5=[[-3,-3,-3],
        [5,0,-3],
        [5,5,-3]]
    k6=[[-3,-3,-3],
        [-3,0,-3],
        [5,5,5]]
    k7=[[-3,-3,-3],
        [-3,0,5],
        [-3,5,5]]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp0=tmp1=tmp2=tmp3=tmp4=tmp5=tmp6=tmp7=0            
            for m in range(3):
                for n in range(3):
                    tmp0=tmp0+k0[m][n]*img[i+m][j+n]
                    tmp1=tmp1+k1[m][n]*img[i+m][j+n]
                    tmp2=tmp2+k2[m][n]*img[i+m][j+n]
                    tmp3=tmp3+k3[m][n]*img[i+m][j+n]
                    tmp4=tmp4+k4[m][n]*img[i+m][j+n]
                    tmp5=tmp5+k5[m][n]*img[i+m][j+n]
                    tmp6=tmp6+k6[m][n]*img[i+m][j+n]
                    tmp7=tmp7+k7[m][n]*img[i+m][j+n]
            tmp=max(tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7)
            #print(kirsch_img[i][j],tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7)
            if tmp>=135:
                kirsch_img[i][j]=0
            else:
                kirsch_img[i][j]=255
    return kirsch_img

kirsch_img=kirsch(gray) 
new_kirsch_img=Image.fromarray(kirsch_img)
new_kirsch_img=new_kirsch_img.convert("L")  
new_kirsch_img.save("new_kirsch_img.bmp")  

def robinson(img):
    robinson_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    k0=[[-1,0,1],
        [-2,0,2],
        [-1,0,1]]
    k1=[[0,1,2],
        [-1,0,1],
        [-2,-1,0]]
    k2=[[1,2,1],
        [0,0,0],
        [-1,-2,-1]]
    k3=[[2,1,0],
        [1,0,-1],
        [0,-1,-2]]
    k4=[[1,0,-1],
        [2,0,-2],
        [1,0,-1]]
    k5=[[0,-1,-2],
        [1,0,-1],
        [2,1,0]]
    k6=[[-1,-2,-1],
        [0,0,0],
        [1,2,1]]
    k7=[[-2,-1,0],
        [-1,0,1],
        [0,1,2]]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp0=tmp1=tmp2=tmp3=tmp4=tmp5=tmp6=tmp7=0            
            for m in range(3):
                for n in range(3):
                    tmp0=tmp0+k0[m][n]*img[i+m][j+n]
                    tmp1=tmp1+k1[m][n]*img[i+m][j+n]
                    tmp2=tmp2+k2[m][n]*img[i+m][j+n]
                    tmp3=tmp3+k3[m][n]*img[i+m][j+n]
                    tmp4=tmp4+k4[m][n]*img[i+m][j+n]
                    tmp5=tmp5+k5[m][n]*img[i+m][j+n]
                    tmp6=tmp6+k6[m][n]*img[i+m][j+n]
                    tmp7=tmp7+k7[m][n]*img[i+m][j+n]
            tmp=max(tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7)
            #print(robinson_img[i][j],tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7)
            if tmp>=43:
                robinson_img[i][j]=0
            else:
                robinson_img[i][j]=255
    return robinson_img

robinson_img=robinson(gray) 
new_robinson_img=Image.fromarray(robinson_img)
new_robinson_img=new_robinson_img.convert("L")  
new_robinson_img.save("new_robinson_img.bmp")

def nevatia(img):
    nevatia_img=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    k0=[[100,100,100,100,100],
        [100,100,100,100,100],
        [0,0,0,0,0],
        [-100,-100,-100,-100,-100],
        [-100,-100,-100,-100,-100]]
    k1=[[100,100,100,100,100],
        [100,100,100,78,-32],
        [100,92,0,-92,-100],
        [32,-78,-100,-100,-100],
        [-100,-100,-100,-100,-100]]
    k2=[[100,100,100,32,-100],
        [100,100,92,-78,-100],
        [100,100,0,-100,-100],
        [100,78,-92,-100,-100],
        [100,-32,-100,-100,-100]]
    k3=[[-100,-100,0,100,100],
        [-100,-100,0,100,100],
        [-100,-100,0,100,100],
        [-100,-100,0,100,100],
        [-100,-100,0,100,100]]
    k4=[[-100,32,100,100,100],
        [-100,-78,92,100,100],
        [-100,-100,0,100,100],
        [-100,-100,-92,78,100],
        [-100,-100,-100,-32,100]]
    k5=[[100,100,100,100,100],
        [-32,78,100,100,100],
        [-100,-92,0,92,100],
        [-100,-100,-100,-78,32],
        [-100,-100,-100,-100,-100]]
    img=cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
    for i in range(height):
        for j in range(width):
            tmp0=tmp1=tmp2=tmp3=tmp4=tmp5=0            
            for m in range(5):
                for n in range(5):
                    tmp0=tmp0+k0[m][n]*img[i+m][j+n]
                    tmp1=tmp1+k1[m][n]*img[i+m][j+n]
                    tmp2=tmp2+k2[m][n]*img[i+m][j+n]
                    tmp3=tmp3+k3[m][n]*img[i+m][j+n]
                    tmp4=tmp4+k4[m][n]*img[i+m][j+n]
                    tmp5=tmp5+k5[m][n]*img[i+m][j+n]
            
            tmp=max(tmp0,tmp1,tmp2,tmp3,tmp4,tmp5)
            #print(tmp,tmp0,tmp1,tmp2,tmp3,tmp4,tmp5)
            if tmp>=12500:
                nevatia_img[i][j]=0
            else:
                nevatia_img[i][j]=255
    return nevatia_img

nevatia_img=nevatia(gray) 
new_nevatia_img=Image.fromarray(nevatia_img)
new_nevatia_img=new_nevatia_img.convert("L")  
new_nevatia_img.save("new_nevatia_img.bmp")
