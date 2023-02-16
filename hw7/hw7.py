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

thinning=np.zeros((64,64),dtype=int)
for i in range(64):
    for j in range(64):
        thinning[i][j]=binary[i*8][j*8]
thinning_img=Image.fromarray(thinning)
thinning_img=thinning_img.convert("L")
thinning_img.save("thinning.bmp")   

def h(b,c,d,e):
    if b==c and (d!=b or e!=b): 
        return "q"
    elif b==c and (d==b and e==b): 
        return "r"
    elif b!=c: 
        return "s"

def f(a1,a2,a3,a4):
    num=0
    if a1==a2==a3==a4=="r":
        return 5
    if a1=="q":
        num=num+1
    if a2=="q":
        num=num+1
    if a3=="q":
        num=num+1
    if a4=="q":
        num=num+1
    return num

while(True):
    #yokoi
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
         
    #PairRelationOperator
    for i in range(64):
        for j in range(64):
            padding[i+1][j+1]=yokoi[i][j]
    for i in range(1,65):
        for j in range(1,65):
            if padding[i][j]==0:#背景
                thinning[i-1][j-1]=0
            elif padding[i][j]==1:#edge
                if padding[i+1][j]!=1 and padding[i-1][j]!=1 and padding[i][j+1]!=1 and padding[i][j-1]!=1:#當該點為edge(yokoi值為1)時，若該點的(上下左右的鄰居皆不為edge)，則該點設為q，反之(上下左右的鄰居也存在edge時)，則該點設為P
                    thinning[i-1][j-1]=2#q
                else:
                    thinning[i-1][j-1]=1#p
            else:#yokoi值為2or3or4or5時，設為q
                thinning[i-1][j-1]=2#q
    
    #ConnectedShrinkOperator
    for i in range(64):
        for j in range(64):
            padding[i+1][j+1]=thinning[i][j]
    for i in range(1,65):
        for j in range(1,65):
            if padding[i][j]!=0:
                padding[i][j]=1
    for i in range(64):
        for j in range(64):
            if thinning[i][j]==1:#每次重新計算yokoi值
                x0=padding[i+1][j+1]
                x1=padding[i+1][j+2]
                x2=padding[i][j+1]
                x3=padding[i+1][j]
                x4=padding[i+2][j+1]
                x5=padding[i+2][j+2]
                x6=padding[i][j+2]
                x7=padding[i][j]
                x8=padding[i+2][j]
                
                a1=h(x0,x1,x6,x2)
                a2=h(x0,x2,x7,x3)
                a3=h(x0,x3,x8,x4)
                a4=h(x0,x4,x5,x1)               
                num=0
                if a1=='q':num=num+1
                if a2=='q':num=num+1
                if a3=='q':num=num+1
                if a4=='q':num=num+1
                
                if num==1:#當只有剛好一個a值結果是1的話(相當於Yokoi中計算q的個數)，才把他變成background值；否則維持原樣。
                    thinning[i][j]=0
                    padding[i+1][j+1]=0
    
            if thinning[i][j]!=0:
                thinning[i][j]=255
    thinning_img=Image.fromarray(thinning)
    thinning_img=thinning_img.convert("L")
    thinning_img.save("thinning.bmp")        
    
    check="true"#當結果不再變化時跳出while迴圈
    for i in range(64):
        for j in range(64):
            if(Downsampling[i][j]!=thinning[i][j]):
                check="false"
    if(check=="true"):
        break
    
    for i in range(64):
        for j in range(64):
            Downsampling[i][j]=thinning[i][j]
            
    
            
      
'''
borderinterior=np.zeros((66,66),dtype=str)#產生border/interior image
for i in range(64):
    for j in range(64):
        if(yokoi[i][j]==5):
            borderinterior[i][j]="i"
        elif(yokoi[i][j]!=5 and yokoi[i][j]!=0):
            borderinterior[i][j]="b"
            
marked=np.zeros((64,64),dtype=int)#產生marked image
for i in range(64):
    for j in range(64):
        if(borderinterior[i][j]=="b"):
            if(borderinterior[i-1][j-1]=="i" or borderinterior[i-1][j]=="i" or borderinterior[i-1][j+1]=="i" or borderinterior[i][j-1]=="i" or borderinterior[i][j+1]=="i" or borderinterior[i+1][j-1]=="i" or borderinterior[i+1][j]=="i" or borderinterior[i+1][j+1]=="i"):
                marked[i][j]=1
            else:
                marked[i][j]=0
'''                














                














