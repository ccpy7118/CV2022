# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:15:36 2022
@author: ccpy
"""
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import measure
import math
# Importing function cv2_imshow necessary for programing in 
img = cv2.imread("C:/Users/ccpy/Desktop/gray.bmp", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("C:/Users/ccpy/Desktop/gray.bmp", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("C:/Users/ccpy/Desktop/gray.bmp", cv2.IMREAD_GRAYSCALE)


i=j=k=0

'''equalize_img = cv2.equalizeHist(img)
cv2.imshow("equal_image", equalize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(equalize_img.ravel(), 256, [0, 255],label= 'equalize image')
'''
height=img.shape[0]
width=img.shape[1]

arr=np.array(img)
#(1)畫直方圖
pic1=np.zeros([height*width])
#print(arr.size)
for i in range(height):
    for j in range(width):
        pic1[k]=(img[i,j])
        k=k+1
n, bins, patches = plt.hist(pic1,bins=256)
plt.show
cv2.imwrite("C:/Users/ccpy/Desktop/1-1.bmp", img)


#(2)畫直方圖

i=j=k=0
pic2=np.zeros([height*width])
pic2 = pic2.astype(int)

for i in range(height):
    for j in range(width):
        pic2[k]=(img[i,j])
    
        pic2[k]=(pic2[k])/3
        k=k+1
n, bins, patches = plt.hist(pic2,bins=256)
plt.show
for i in range(height):
    for j in range(width):
        img2[i,j]=img[i,j]/3
cv2.imwrite("C:/Users/ccpy/Desktop/2-1.bmp", img2)


'''
im = Image.open("C:/Users/ccpy/Desktop/gray.bmp")
lenaHistogram = np.zeros(256,dtype=np.int32)
row,col = im.size
for i in range(row):
    for j in range(col):
        tmp = im.getpixel((i,j))
        lenaHistogram[tmp]=lenaHistogram[tmp]+1

a = np.arange(256)
plt.bar(a,lenaHistogram)
plt.savefig("C:/Users/ccpy/Desktop/bmphistogram.png")
'''
#(3)histogram equalization
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import measure
import math
i=j=k=t=0
equalization=np.zeros([height*width])
equalization = equalization.astype(int)

for i in range(height):
    for j in range(width):
        equalization[k]=(img[i,j])/3
        k=k+1
        
count=np.zeros([256])
count = count.astype(int)   
k=0     
for k in range(height*width):
    count[equalization[k]]=count[equalization[k]]+1

k=0
for k in range(8,82,1):
    count[k]=count[k]+count[k-1]
#參考https://en.wikipedia.org/wiki/Histogram_equalization
k=0
cdf=np.zeros([256])
for k in range(8,82,1):
    cdf[k]=round((count[k]-1)/(height*width-1)*255)
    
i=j=k=0     
for k in range(height*width):
        equalization[k]=cdf[equalization[k]]
   
n, bins, patches = plt.hist(equalization,bins=256)
plt.show
for i in range(height):
    for j in range(width):
        img3[i,j]=img2[i,j]
        img3[i,j]=cdf[img3[i,j]]
cv2.imwrite("C:/Users/ccpy/Desktop/3-1.bmp", img3)

