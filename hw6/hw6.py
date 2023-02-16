import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import measure

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
Downsampling =np.zeros((64,64),dtype=int)
for i in range(64):
    for j in range(64):
        Downsampling[i][j]=binary[i*8][j*8]
Downsampling_img=Image.fromarray(Downsampling)
Downsampling_img=Downsampling_img.convert("L")
Downsampling_img.save("Downsampling.bmp")   

def h(b,c,d,e):
    if b==c and (d!=b or e!=b): 
        return "q"
    elif b==c and (d==b and e==b): 
        return "r"
    elif b!=c: 
        return "s"

def f(a1,a2,a3,a4):
    count=0
    if a1==a2==a3==a4=="r":
        return 5
    if a1=="q":
        count=count+1
    if a2=="q":
        count=count+1
    if a3=="q":
        count=count+1
    if a4=="q":
        count=count+1
    return count

padding=np.zeros((66,66),dtype=int)
for i in range(64):
    for j in range(64):
        padding[i+1][j+1]=Downsampling[i][j]

yokoi=np.zeros((64,64),dtype=int)
for i in range(66):
    for j in range(66):        
        if(padding[i][j]==255 and i-1>=0 and i+1<=65 and j-1>=0 and j+1<=65):
            x0=padding[i][j]
            x1=padding[i][j+1]
            x2=padding[i-1][j]
            x3=padding[i][j-1]
            x4=padding[i+1][j]
            x5=padding[i+1][j+1]
            x6=padding[i-1][j+1]
            x7=padding[i-1][j-1]
            x8=padding[i+1][j-1]
            a1=h(x0,x1,x6,x2)
            a2=h(x0,x2,x7,x3)
            a3=h(x0,x3,x8,x4)
            a4=h(x0,x4,x5,x1)
            yokoi[i-1][j-1]=f(a1,a2,a3,a4)

with open('yokoi.txt', 'w') as f:
    for i in range(64):
        for j in range(64):
            if(yokoi[i][j]==0):
                f.write(" ")
            elif(yokoi[i][j]!=0):
                f.write(str(yokoi[i][j]))
        f.write("\n")













