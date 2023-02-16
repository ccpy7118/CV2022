# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

 
#include<cstdio>
#include<cstdlib>
#include<cmath>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import measure

# Importing function cv2_imshow necessary for programing in 
img = cv2.imread("lena.bmp")
img1 = cv2.imread("lena.bmp")

height=img1.shape[0]
width=img1.shape[1]
#(1)二值化
for i in range(height):
    for j in range(width):
        if img1[i,j,0]<128:
            img1[i,j,0]=0
            img1[i,j,1]=0
            img1[i,j,2]=0

        elif img1[i,j,0]>=128:
            img1[i,j,0]=255
            img1[i,j,1]=255
            img1[i,j,2]=255


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); # 轉換前，都先將圖片轉換成灰階色彩
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); # 轉換前，都先將圖片轉換成灰階色彩
'''
for i in range(height):
    for j in range(width):
        print(img[i][j])
        print(gray1[i][j])
'''

#ret, output1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("C:/Users/ccpy/Desktop/img.bmp", img)
cv2.imwrite("C:/Users/ccpy/Desktop/img1.bmp", img1)
cv2.imwrite("C:/Users/ccpy/Desktop/img7.bmp", gray)
cv2.imwrite("C:/Users/ccpy/Desktop/gray.bmp", gray1)

#plt.hist(gray.ravel(), 256, [0, 256])
#plt.show()
k=0

#(2)畫直方圖
height=img.shape[0]
width=img.shape[1]

pic = [0]*(height*width)
arr=np.array(gray)

pic = [0]*(height*width)
gray1=np.array(gray1)
print(arr.size)
for i in range(height):
    for j in range(width):
        pic[k]=(gray[i,j])
    
        #pic[k]=(pic[k])/3
        k=k+1
n, bins, patches = plt.hist(pic,bins=256)
plt.show
#print(gray[i,j])


for i in range(height):
    for j in range(width):
        if gray1[i,j]<128:
            gray1[i,j]=0
        elif gray1[i,j]>=128:
            gray1[i,j]=255
            
        #print(gray[i,j])
labe=np.zeros([height+1,width+1])
labe = labe.astype(int)

#將值為255的pixel標編號1
for i in range(height):
    for j in range(width):
        if (gray1[i,j])==255:
            labe[i,j]=1

'''for i in range (1):
    for j in range (100):
        print(labe[i,j])
'''
#將編號為1的pixel標不同的編號
i=0
j=0       
p=1
for i in range(height):
    for j in range(width):
        if (labe[i,j])==1:
            labe[i,j]=p
            p=p+1
            
print(labe[-1,0])

#合併區域

i=0
j=0
m=0
n=0
t=0
tmp=0
for t in range(50):  
    for i in range(height):
        for j in range(width):
            if (labe[i,j]!=0):
                tmp=labe[i,j]
                if (labe[i,j-1])!=0 and labe[i-1,j]!=0 :
                    if((labe[i,j-1]) >= (labe[i-1,j])):
                        smaller = (labe[i-1,j])
                        if(tmp>smaller):
                            labe[i,j]=smaller
                        elif(tmp<=smaller):
                            labe[i,j]=tmp
                    elif((labe[i,j-1]) < (labe[i-1,j])):
                        smaller = (labe[i,j-1])
                        if(tmp>smaller):
                            labe[i,j]=smaller
                        elif(tmp<=smaller):
                            labe[i,j]=tmp
                elif(labe[i,j-1])==0 and labe[i-1,j]!=0:
                    if(labe[i-1,j]<labe[i,j]):
                        labe[i,j]=labe[i-1,j]
                elif(labe[i,j-1])!=0 and labe[i-1,j]==0:
                    if(labe[i,j-1]<labe[i,j]):
                        labe[i,j]=labe[i,j-1]
    for m in range(height,-1,-1):
        for n in range(width,-1,-1):
            if (labe[m,n]!=0):
                tmp=labe[m,n]
                if (labe[m,n+1])!=0 and labe[m+1,n]!=0 :
                    if((labe[m,n+1]) >= (labe[m+1,n])):
                        smaller = (labe[m+1,n])
                        if(tmp>smaller):
                            labe[m,n]=smaller
                        elif(tmp<=smaller):
                            labe[m,n]=tmp
                    elif((labe[m,n+1]) < (labe[m+1,n])):
                        smaller = (labe[m,n+1])
                        if(tmp>smaller):
                            labe[m,n]=smaller
                        elif(tmp<=smaller):
                            labe[m,n]=tmp
                elif(labe[m,n+1])==0 and labe[m+1,n]!=0:
                    if(labe[m+1,n]<labe[m,n]):
                        labe[m,n]=labe[m+1,n]
                elif(labe[m,n+1])!=0 and labe[m+1,n]==0:
                    if(labe[m,n+1]<labe[m,n]):
                        labe[m,n]=labe[m,n+1]

#看總共幾個區域
i=j=0    
for i in range (height):
    for j in range (width):
        #print(labe[i,j])
        if(labe[i,j]!=0)and labe[i+1,j]!=labe[i,j] and labe[i,j+1]!=labe[i,j] and labe[i-1,j]!=labe[i,j] and labe[i,j-1]!=labe[i,j]:
            t+=1
print(t)
#決定大於500的區域
areasize=np.zeros([height*width])
areasize = areasize.astype(int)
k=0
for i in range (height):
    for j in range (width):
        if labe[i,j]!=0 :
            k=labe[i,j]
            areasize[k]=areasize[k]+1
j=0
for i in range(height*width):
    if(areasize[i]>=500):
        j=j+1
        print(i,":",areasize[i])
print(j)

'''
k=0
while (1>0):
    if j<=0:
        break
    elif j>0:
        color = (0,0,255)
        leftx=lefty=550
        rightx=righty=0
        sumx=sumy=times=0
        for k in range(height*width):
            if(areasize[k]>=500):
                for i in range (height):
                    for j in range (width):
                        if labe[i,j]==1:
                            sumy=sumy+i
                            sumx=sumx+j
                            times=times+1
                            if i<lefty:
                                lefty=i
                            if j<leftx:
                                leftx=j
                            if i>righty:
                                righty=i
                            if j>rightx:
                                rightx=j
                sumy=int(sumy/times)
                sumx=int(sumx/times)
            print(sumx,sumy)
            print(leftx,lefty,rightx, righty)
            cv2.drawMarker(img1,(sumx,sumy),color,markerType=1)
            cv2.rectangle(img1, (leftx,lefty), (rightx, righty), color, 2)
'''            

#畫圖
color = (0,0,255)
leftx=lefty=550
rightx=righty=0
sumx=sumy=times=0
while(1>0):
    for i in range (height):
        for j in range (width):
            if labe[i,j]==1:
                sumy=sumy+i
                sumx=sumx+j
                times=times+1
                if i<lefty:
                    lefty=i
                if j<leftx:
                    leftx=j
                if i>righty:
                    righty=i
                if j>rightx:
                    rightx=j
    sumy=int(sumy/times)
    sumx=int(sumx/times)
    break
print(sumx,sumy)
print(leftx,lefty,rightx, righty)
cv2.drawMarker(img1,(sumx,sumy),color,markerType=0)
cv2.rectangle(img1, (leftx,lefty), (rightx, righty), color, 2)

leftx=lefty=550
rightx=righty=0
sumx=sumy=times=0
while(1>0):
    for i in range (height):
        for j in range (width):
            if labe[i,j]==73:
                sumy=sumy+i
                sumx=sumx+j
                times=times+1
                if i<lefty:
                    lefty=i
                if j<leftx:
                    leftx=j
                if i>righty:
                    righty=i
                if j>rightx:
                    rightx=j
    sumy=int(sumy/times)
    sumx=int(sumx/times)
    break
print(sumx,sumy)
print(leftx,lefty,rightx, righty)
cv2.drawMarker(img1,(sumx,sumy),color,markerType=0)
cv2.rectangle(img1, (leftx,lefty), (rightx, righty), color, 2)


leftx=lefty=550
rightx=righty=0
sumx=sumy=times=0
while(1>0):
    for i in range (height):
        for j in range (width):
            if labe[i,j]==26099:
                sumy=sumy+i
                sumx=sumx+j
                times=times+1
                if i<lefty:
                    lefty=i
                if j<leftx:
                    leftx=j
                if i>righty:
                    righty=i
                if j>rightx:
                    rightx=j
    sumy=int(sumy/times)
    sumx=int(sumx/times)
    break
print(sumx,sumy)
print(leftx,lefty,rightx, righty)
cv2.drawMarker(img1,(sumx,sumy),color,markerType=0)
cv2.rectangle(img1, (leftx,lefty), (rightx, righty), color, 2)


leftx=lefty=550
rightx=righty=0
sumx=sumy=times=0
while(1>0):
    for i in range (height):
        for j in range (width):
            if labe[i,j]==66809:
                sumy=sumy+i
                sumx=sumx+j
                times=times+1
                if i<lefty:
                    lefty=i
                if j<leftx:
                    leftx=j
                if i>righty:
                    righty=i
                if j>rightx:
                    rightx=j
    sumy=int(sumy/times)
    sumx=int(sumx/times)
    break
print(sumx,sumy)
print(leftx,lefty,rightx, righty)
cv2.drawMarker(img1,(sumx,sumy),color,markerType=0)
cv2.rectangle(img1, (leftx,lefty), (rightx, righty), color, 2)

leftx=lefty=550
rightx=righty=0
sumx=sumy=times=0
while(1>0):
    for i in range (height):
        for j in range (width):
            if labe[i,j]==107770:
                sumy=sumy+i
                sumx=sumx+j
                times=times+1
                if i<lefty:
                    lefty=i
                if j<leftx:
                    leftx=j
                if i>righty:
                    righty=i
                if j>rightx:
                    rightx=j
    sumy=int(sumy/times)
    sumx=int(sumx/times)
    break
print(sumx,sumy)
print(leftx,lefty,rightx, righty)
cv2.drawMarker(img1,(sumx,sumy),color,markerType=0)
cv2.rectangle(img1, (leftx,lefty), (rightx, righty), color, 2)



cv2.imshow('Output', img1)
cv2.imwrite("C:/Users/ccpy/Desktop/imglabel.bmp", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()