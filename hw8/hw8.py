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
'''
def snr_func(img,img_noise):
    img=img.copy()
    img_noise=img_noise.copy()
    tmp=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp=img[i][j]+tmp
    average=tmp/262144
    tmp=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp=tmp+(math.pow((img[i][j]-average),2))
    vs=tmp/262144
    tmp=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp=tmp+(img_noise[i][j]-img[i][j])
    average_noise=tmp/262144
    tmp=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp=tmp+(math.pow((img_noise[i][j]-img[i][j]-average_noise),2))
    vn=tmp/262144
    vs=math.sqrt(vs)
    vn=math.sqrt(vn)
    tmp=(vs/vn)
    snr=20*math.log(tmp,10)
    return snr
'''
def snr_func(img,img_noise):
    row,col=img.shape
    height=img.shape[0]
    width=img.shape[1]
    img_noise=img_noise/255
    img=img/255

    u=np.mean(img)
    vs=0
    for i in range(height):
        for j in range(width):
            vs=vs+(math.pow(img[i][j]-u,2))/(height*width)

    un=0
    for i in range(height):
        for j in range(width):
            un=un+(img_noise[i][j]-img[i][j])
    un=un/(height*width)

    vn=0
    for i in range(height):
        for j in range(width):
            vn=vn+(math.pow(img_noise[i][j]-img[i][j]-un,2))
    vn=vn/(height*width)

    return 20*math.log10(math.sqrt(vs)/math.sqrt(vn))

def getgaussiannoise_image_10(img):
    gaussiannoise_image_10=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    noisepixel=img+10*np.random.normal(0,1,(height,width))
    gaussiannoise_image_10=noisepixel
    return gaussiannoise_image_10
'''
im=Image.open('gray.bmp')
img = np.asarray(im)
row,col=img.shape
img_ori=np.zeros((row,col),dtype=int)
img_ori[0:row,0:col]=img[0:row,0:col]

img.flags.writeable = True

img=img+10*np.random.normal(0,1,(row,col))
im=Image.fromarray(img)
im.show()
im=im.convert("L")
im.save('lena_Gaussian10.bmp',format='BMP')
snr=snr_func(img_ori,img)
#print(snr)

img[0:row,0:col]=img_ori[0:row,0:col]
img=img+30*np.random.normal(0,1,(row,col))
im=Image.fromarray(img)
im.show()
im=im.convert("L")
im.save('lena_Gaussian30.bmp',format='BMP')
snr=snr_func(img_ori,img)
#print(snr)
'''
def getgaussiannoise_image_30(img):
    gaussiannoise_image_30=img.copy()
    height=img.shape[0]
    width=img.shape[1]
    noisepixel=img+30*np.random.normal(0,1,(height,width))
    gaussiannoise_image_30=noisepixel
    return gaussiannoise_image_30

def getsaltandpepper_image_005(img):
    saltandpepper_image_005 = img.copy()
    for c in range(img.shape[0]):
        for r in range(img.shape[1]):
            random_value=random.uniform(0,1)
            if (random_value<=0.05):
                saltandpepper_image_005[c][r]=0
            elif(random_value>=(1-0.05)):
                saltandpepper_image_005[c][r]=255
            else:
                saltandpepper_image_005[c][r]=img[c][r]
    return saltandpepper_image_005 
'''
im=Image.open('lena.bmp')
img = np.asarray(im)
row,col=img.shape
img_ori=np.zeros((row,col),dtype=int)
img_ori[0:row,0:col]=img[0:row,0:col]

img.flags.writeable = True

noise = np.random.uniform(0.0,1.0,(row,col))

for i in range(row):
    for j in range(col):
        if noise[i][j]>0.95:
            img[i][j]=255
        elif noise[i][j]<0.05:
            img[i][j]=0

snr=snr_func(img_ori,img)
#print(snr)

im=Image.fromarray(img)
im.show()
im=im.convert("L")
im.save('lena_salt_and_pepper0.05.bmp',format='BMP')

img[0:row,0:col]=img_ori[0:row,0:col]

noise = np.random.uniform(0.0,1.0,(row,col))

for i in range(row):
    for j in range(col):
        if noise[i][j]>0.9:
            img[i][j]=255
        elif noise[i][j]<0.1:
            img[i][j]=0

snr=snr_func(img_ori,img)
#print(snr)

im=Image.fromarray(img)
im.show()
im=im.convert("L")
im.save('lena_salt_and_pepper0.1.bmp',format='BMP')
'''
def getsaltandpepper_image_01(img):
    saltandpepper_image_01 = img.copy()
    for c in range(img.shape[0]):
        for r in range(img.shape[1]):
            random_value=random.uniform(0,1)
            if (random_value<=0.1):
                saltandpepper_image_01[c][r]=0
            elif(random_value>=(1-0.1)):
                saltandpepper_image_01[c][r]=255
            else:
                saltandpepper_image_01[c][r]=img[c][r]
    return saltandpepper_image_01           

gaussiannoise_10=getgaussiannoise_image_10(gray) 
gaussiannoise_10_img=Image.fromarray(gaussiannoise_10)
gaussiannoise_10_img=gaussiannoise_10_img.convert("L")  
gaussiannoise_10_img.save("gaussiannoise_10_img.bmp")        
print('gaussiannoise_10:',snr_func(gray,gaussiannoise_10))
            
gaussiannoise_30=getgaussiannoise_image_30(gray) 
gaussiannoise_30_img=Image.fromarray(gaussiannoise_30)
gaussiannoise_30_img=gaussiannoise_30_img.convert("L")  
gaussiannoise_30_img.save("gaussiannoise_30_img.bmp")  
print('gaussiannoise_30:',snr_func(gray,gaussiannoise_30))
              
            
saltandpepper_005=getsaltandpepper_image_005(gray) 
saltandpepper_005_img=Image.fromarray(saltandpepper_005)
saltandpepper_005_img=saltandpepper_005_img.convert("L")  
saltandpepper_005_img.save("saltandpepper_005_img.bmp")  
print('saltandpepper_005_img:',snr_func(gray,saltandpepper_005))
            
            
saltandpepper_01=getsaltandpepper_image_01(gray) 
saltandpepper_01_img=Image.fromarray(saltandpepper_01)
saltandpepper_01_img=saltandpepper_01_img.convert("L")  
saltandpepper_01_img.save("saltandpepper_01_img.bmp")  
print('saltandpepper_01_img:',snr_func(gray,saltandpepper_01))

            
def filter_33(img):
    filter_image_33 = img.copy()
    height=img.shape[0]
    width=img.shape[1]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for c in range(height):
        for r in range(width):
            filter_image_33[c][r]=int((int(img[c][r])+int(img[c][r+1])+int(img[c][r+2])+int(img[c+1][r])+int(img[c+1][r+1])+int(img[c+1][r+2])+int(img[c+2][r])+int(img[c+2][r+1])+int(img[c+2][r+2]))/9)
    return filter_image_33            
            
filter33_gaussiannoise_10=filter_33(gaussiannoise_10) 
filter33_gaussiannoise_10_img=Image.fromarray(filter33_gaussiannoise_10)
filter33_gaussiannoise_10_img=filter33_gaussiannoise_10_img.convert("L")  
filter33_gaussiannoise_10_img.save("filter33_gaussiannoise_10_img.bmp") 
print('filter33_gaussiannoise_10_img:',snr_func(gray,filter33_gaussiannoise_10))


filter33_gaussiannoise_30=filter_33(gaussiannoise_30) 
filter33_gaussiannoise_30_img=Image.fromarray(filter33_gaussiannoise_30)
filter33_gaussiannoise_30_img=filter33_gaussiannoise_30_img.convert("L")  
filter33_gaussiannoise_30_img.save("filter33_gaussiannoise_30_img.bmp")      
print('filter33_gaussiannoise_30_img:',snr_func(gray,filter33_gaussiannoise_30))

filter33_saltandpepper_005=filter_33(saltandpepper_005) 
filter33_saltandpepper_005_img=Image.fromarray(filter33_saltandpepper_005)
filter33_saltandpepper_005_img=filter33_saltandpepper_005_img.convert("L")  
filter33_saltandpepper_005_img.save("filter33_saltandpepper_005_img.bmp")      
print('filter33_saltandpepper_005_img:',snr_func(gray,filter33_saltandpepper_005))

filter33_saltandpepper_01=filter_33(saltandpepper_01) 
filter33_saltandpepper_01_img=Image.fromarray(filter33_saltandpepper_01)
filter33_saltandpepper_01_img=filter33_saltandpepper_01_img.convert("L")  
filter33_saltandpepper_01_img.save("filter33_saltandpepper_01_img.bmp")                   
print('filter33_saltandpepper_01_img:',snr_func(gray,filter33_saltandpepper_01))

def filter_55(img):
    filter_image_55 = img.copy()
    height=img.shape[0]
    width=img.shape[1]
    img=cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
    for c in range(height):
        for r in range(width):
            filter_image_55[c][r]=int((int(img[c][r])+int(img[c][r+1])+int(img[c][r+2])+int(img[c][r+3])+int(img[c][r+4])+
                                       int(img[c+1][r])+int(img[c+1][r+1])+int(img[c+1][r+2])+int(img[c+1][r+3])+int(img[c+1][r+4])+
                                       int(img[c+2][r])+int(img[c+2][r+1])+int(img[c+2][r+2])+int(img[c+2][r+3])+int(img[c+2][r+4])+
                                       int(img[c+3][r])+int(img[c+3][r+1])+int(img[c+3][r+2])+int(img[c+3][r+3])+int(img[c+3][r+4])+
                                       int(img[c+4][r])+int(img[c+4][r+1])+int(img[c+4][r+2])+int(img[c+4][r+3])+int(img[c+4][r+4]))/25)
    return filter_image_55            
            
filter55_gaussiannoise_10=filter_55(gaussiannoise_10) 
filter55_gaussiannoise_10_img=Image.fromarray(filter55_gaussiannoise_10)
filter55_gaussiannoise_10_img=filter55_gaussiannoise_10_img.convert("L")  
filter55_gaussiannoise_10_img.save("filter55_gaussiannoise_10_img.bmp") 
print('filter55_gaussiannoise_10_img:',snr_func(gray,filter55_gaussiannoise_10))

filter55_gaussiannoise_30=filter_55(gaussiannoise_30) 
filter55_gaussiannoise_30_img=Image.fromarray(filter55_gaussiannoise_30)
filter55_gaussiannoise_30_img=filter55_gaussiannoise_30_img.convert("L")  
filter55_gaussiannoise_30_img.save("filter55_gaussiannoise_30_img.bmp")      
print('filter55_gaussiannoise_30_img:',snr_func(gray,filter55_gaussiannoise_30))

filter55_saltandpepper_005=filter_55(saltandpepper_005) 
filter55_saltandpepper_005_img=Image.fromarray(filter55_saltandpepper_005)
filter55_saltandpepper_005_img=filter55_saltandpepper_005_img.convert("L")  
filter55_saltandpepper_005_img.save("filter55_saltandpepper_005_img.bmp")      
print('filter55_saltandpepper_005_img:',snr_func(gray,filter55_saltandpepper_005))

filter55_saltandpepper_01=filter_55(saltandpepper_01) 
filter55_saltandpepper_01_img=Image.fromarray(filter55_saltandpepper_01)
filter55_saltandpepper_01_img=filter55_saltandpepper_01_img.convert("L")  
filter55_saltandpepper_01_img.save("filter55_saltandpepper_01_img.bmp")                    
print('filter55_saltandpepper_01_img:',snr_func(gray,filter55_saltandpepper_01))
       
def bubblesort_median9(data):
    n = len(data)
    for i in range(n-2):                   # 有 n 個資料長度，但只要執行 n-1 次
        for j in range(n-i-1):             # 從第1個開始比較直到最後一個還沒到最終位置的數字 
            if data[j] > data[j+1]:        # 比大小然後互換
                data[j], data[j+1] = data[j+1], data[j]
    return data[4]
def median_filter_33(img):
    data =np.zeros((9),dtype=int)
    median_filter_image_33 = img.copy()
    height=img.shape[0]
    width=img.shape[1]
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    for c in range(height):
        for r in range(width):
            k=0
            for i in range (3):
                for j in range(3):
                    data[k]=img[c+i][r+j]
                    k=k+1
            median_filter_image_33[c][r]=bubblesort_median9(data)
    return median_filter_image_33              
            
median_filter_33_gaussiannoise_10=median_filter_33(gaussiannoise_10) 
median_filter_33_gaussiannoise_10_img=Image.fromarray(median_filter_33_gaussiannoise_10)
median_filter_33_gaussiannoise_10_img=median_filter_33_gaussiannoise_10_img.convert("L")  
median_filter_33_gaussiannoise_10_img.save("median_filter_33_gaussiannoise_10_img.bmp") 
print('median_filter_33_gaussiannoise_10_img:',snr_func(gray,median_filter_33_gaussiannoise_10))

median_filter_33_gaussiannoise_30=median_filter_33(gaussiannoise_30) 
median_filter_33_gaussiannoise_30_img=Image.fromarray(median_filter_33_gaussiannoise_30)
median_filter_33_gaussiannoise_30_img=median_filter_33_gaussiannoise_30_img.convert("L")  
median_filter_33_gaussiannoise_30_img.save("median_filter_33_gaussiannoise_30_img.bmp")      
print('median_filter_33_gaussiannoise_30_img:',snr_func(gray,median_filter_33_gaussiannoise_30))

median_filter_33_saltandpepper_005=median_filter_33(saltandpepper_005) 
median_filter_33_saltandpepper_005_img=Image.fromarray(median_filter_33_saltandpepper_005)
median_filter_33_saltandpepper_005_img=median_filter_33_saltandpepper_005_img.convert("L")  
median_filter_33_saltandpepper_005_img.save("median_filter_33_saltandpepper_005_img.bmp")      
print('median_filter_33_saltandpepper_005_img:',snr_func(gray,median_filter_33_saltandpepper_005))

median_filter_33_saltandpepper_01=median_filter_33(saltandpepper_01) 
median_filter_33_saltandpepper_01_img=Image.fromarray(median_filter_33_saltandpepper_01)
median_filter_33_saltandpepper_01_img=median_filter_33_saltandpepper_01_img.convert("L")  
median_filter_33_saltandpepper_01_img.save("median_filter_33_saltandpepper_01_img.bmp")            
print('median_filter_33_saltandpepper_01_img:',snr_func(gray,median_filter_33_saltandpepper_01))

            
def bubblesort_median25(data):
    n = len(data)
    for i in range(n-2):                   # 有 n 個資料長度，但只要執行 n-1 次
        for j in range(n-i-1):             # 從第1個開始比較直到最後一個還沒到最終位置的數字 
            if data[j] > data[j+1]:        # 比大小然後互換
                data[j], data[j+1] = data[j+1], data[j]
    return data[12]
def median_filter_55(img):
    data =np.zeros((25),dtype=int)
    median_filter_image_55 = img.copy()
    height=img.shape[0]
    width=img.shape[1]
    img=cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
    for c in range(height):
        for r in range(width):
            k=0
            for i in range (5):
                for j in range(5):
                    data[k]=img[c+i][r+j]
                    k=k+1
            median_filter_image_55[c][r]=bubblesort_median25(data)
    return median_filter_image_55              
                        
median_filter_55_gaussiannoise_10=median_filter_55(gaussiannoise_10) 
median_filter_55_gaussiannoise_10_img=Image.fromarray(median_filter_55_gaussiannoise_10)
median_filter_55_gaussiannoise_10_img=median_filter_55_gaussiannoise_10_img.convert("L")  
median_filter_55_gaussiannoise_10_img.save("median_filter_55_gaussiannoise_10_img.bmp") 
print('median_filter_55_gaussiannoise_10_img:',snr_func(gray,median_filter_55_gaussiannoise_10))

median_filter_55_gaussiannoise_30=median_filter_55(gaussiannoise_30) 
median_filter_55_gaussiannoise_30_img=Image.fromarray(median_filter_55_gaussiannoise_30)
median_filter_55_gaussiannoise_30_img=median_filter_55_gaussiannoise_30_img.convert("L")  
median_filter_55_gaussiannoise_30_img.save("median_filter_55_gaussiannoise_30_img.bmp")      
print('median_filter_55_gaussiannoise_30_img:',snr_func(gray,median_filter_55_gaussiannoise_30))

median_filter_55_saltandpepper_005=median_filter_55(saltandpepper_005) 
median_filter_55_saltandpepper_005_img=Image.fromarray(median_filter_55_saltandpepper_005)
median_filter_55_saltandpepper_005_img=median_filter_55_saltandpepper_005_img.convert("L")  
median_filter_55_saltandpepper_005_img.save("median_filter_55_saltandpepper_005_img.bmp")      
print('median_filter_55_saltandpepper_005_img:',snr_func(gray,median_filter_55_saltandpepper_005))

median_filter_55_saltandpepper_01=median_filter_55(saltandpepper_01) 
median_filter_55_saltandpepper_01_img=Image.fromarray(median_filter_55_saltandpepper_01)
median_filter_55_saltandpepper_01_img=median_filter_55_saltandpepper_01_img.convert("L")  
median_filter_55_saltandpepper_01_img.save("median_filter_55_saltandpepper_01_img.bmp")                  
print('median_filter_55_saltandpepper_01_img:',snr_func(gray,median_filter_55_saltandpepper_01))
            
#erosion_gray
img = cv2.imread("lena.bmp")
height=img.shape[0]
width=img.shape[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.bmp", gray)
def erosion_func(gray): 
    gray.flags.writeable = True
    kernel=np.array([[0,1,1,1,0],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [0,1,1,1,0]])
    kernel_sum=0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel_sum=kernel_sum+kernel[i][j]
    erosion=np.zeros((height,width),dtype=int)
    for i in range(height):
        for j in range(width):
            tmp=gray[i][j]       
            for x in range(-2,3,1):
                for y in range(-2,3,1):
                    if (0<=x+i and x+i<height and 0<=y+j and y+j<width):
                        if kernel[x+2][y+2]==1 and gray[i+x][j+y]-kernel[x+2][y+2]<tmp:
                            tmp=gray[i+x][j+y]-kernel[x+2][y+2]
                            erosion[i][j]=tmp
                    else:
                        tmp=0
                        erosion[i][j]=tmp
    return erosion 

def dilation_func(gray):
    gray.flags.writeable = True
    kernel=np.array([[0,1,1,1,0],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [0,1,1,1,0]])
    dilation=np.zeros((height,width),dtype=int)

    for i in range(height):
        for j in range(width):
            tmp=0
            for x in range(-2,3,1):
                for y in range(-2,3,1):
                    if (0<=x+i and x+i<height and 0<=y+j and y+j<width):
                        if kernel[x+2][y+2]==1 and gray[i+x][j+y]+kernel[x+2][y+2]>tmp:
                            tmp=gray[i+x][j+y]+kernel[x+2][y+2]
                            dilation[i][j]=tmp
    return dilation

gaussiannoise_10_erosion1=erosion_func(gaussiannoise_10)    
gaussiannoise_10_dilation1=dilation_func(gaussiannoise_10_erosion1)
gaussiannoise_10_dilation2=dilation_func(gaussiannoise_10_dilation1)
gaussiannoise_10_erosion2=erosion_func(gaussiannoise_10_dilation2) 
gaussiannoise_10_erosion2_img=Image.fromarray(gaussiannoise_10_erosion2)
gaussiannoise_10_erosion2_img=gaussiannoise_10_erosion2_img.convert("L")
gaussiannoise_10_erosion2_img.save("gaussiannoise_10_opening_closing_img.bmp")
print('gaussiannoise_10_opening_closing_img:',snr_func(gray,gaussiannoise_10_erosion2))

gaussiannoise_30_erosion1=erosion_func(gaussiannoise_30)    
gaussiannoise_30_dilation1=dilation_func(gaussiannoise_30_erosion1)
gaussiannoise_30_dilation2=dilation_func(gaussiannoise_30_dilation1)
gaussiannoise_30_erosion2=erosion_func(gaussiannoise_30_dilation2) 
gaussiannoise_30_erosion2_img=Image.fromarray(gaussiannoise_30_erosion2)
gaussiannoise_30_erosion2_img=gaussiannoise_30_erosion2_img.convert("L")
gaussiannoise_30_erosion2_img.save("gaussiannoise_30_opening_closing_img.bmp")
print('gaussiannoise_30_opening_closing_img:',snr_func(gray,gaussiannoise_30_erosion2))

saltandpepper_005_erosion1=erosion_func(saltandpepper_005)    
saltandpepper_005_dilation1=dilation_func(saltandpepper_005_erosion1)
saltandpepper_005_dilation2=dilation_func(saltandpepper_005_dilation1)
saltandpepper_005_erosion2=erosion_func(saltandpepper_005_dilation2) 
saltandpepper_005_erosion2_img=Image.fromarray(saltandpepper_005_erosion2)
saltandpepper_005_erosion2_img=saltandpepper_005_erosion2_img.convert("L")
saltandpepper_005_erosion2_img.save("saltandpepper_005_opening_closing_img.bmp")
print('saltandpepper_005_opening_closing_img:',snr_func(gray,saltandpepper_005_erosion2))

saltandpepper_01_erosion1=erosion_func(saltandpepper_01)    
saltandpepper_01_dilation1=dilation_func(saltandpepper_01_erosion1)
saltandpepper_01_dilation2=dilation_func(saltandpepper_01_dilation1)
saltandpepper_01_erosion2=erosion_func(saltandpepper_01_dilation2) 
saltandpepper_01_erosion2_img=Image.fromarray(saltandpepper_01_erosion2)
saltandpepper_01_erosion2_img=saltandpepper_01_erosion2_img.convert("L")
saltandpepper_01_erosion2_img.save("saltandpepper_01_opening_closing_img.bmp")
print('saltandpepper_01_opening_closing_img:',snr_func(gray,saltandpepper_01_erosion2))

gaussiannoise_10_dilation1=dilation_func(gaussiannoise_10)    
gaussiannoise_10_erosion1=erosion_func(gaussiannoise_10_dilation1)
gaussiannoise_10_erosion2=erosion_func(gaussiannoise_10_erosion1)
gaussiannoise_10_dilation2=dilation_func(gaussiannoise_10_erosion2) 
gaussiannoise_10_dilation2_img=Image.fromarray(gaussiannoise_10_dilation2)
gaussiannoise_10_dilation2_img=gaussiannoise_10_dilation2_img.convert("L")
gaussiannoise_10_dilation2_img.save("gaussiannoise_10_closing_opening_img.bmp")
print('gaussiannoise_10_closing_opening_img:',snr_func(gray,gaussiannoise_10_dilation2))

gaussiannoise_30_dilation1=dilation_func(gaussiannoise_30)    
gaussiannoise_30_erosion1=erosion_func(gaussiannoise_30_dilation1)
gaussiannoise_30_erosion2=erosion_func(gaussiannoise_30_erosion1)
gaussiannoise_30_dilation2=dilation_func(gaussiannoise_30_erosion2) 
gaussiannoise_30_dilation2_img=Image.fromarray(gaussiannoise_30_dilation2)
gaussiannoise_30_dilation2_img=gaussiannoise_30_dilation2_img.convert("L")
gaussiannoise_30_dilation2_img.save("gaussiannoise_30_closing_opening_img.bmp")
print('gaussiannoise_30_closing_opening_img:',snr_func(gray,gaussiannoise_30_dilation2))

saltandpepper_005_dilation1=dilation_func(saltandpepper_005)    
saltandpepper_005_erosion1=erosion_func(saltandpepper_005_dilation1)
saltandpepper_005_erosion2=erosion_func(saltandpepper_005_erosion1)
saltandpepper_005_dilation2=dilation_func(saltandpepper_005_erosion2) 
saltandpepper_005_dilation2_img=Image.fromarray(saltandpepper_005_dilation2)
saltandpepper_005_dilation2_img=saltandpepper_005_dilation2_img.convert("L")
saltandpepper_005_dilation2_img.save("saltandpepper_005_closing_opening_img.bmp")
print('saltandpepper_005_closing_opening_img:',snr_func(gray,saltandpepper_005_dilation2))

saltandpepper_01_dilation1=dilation_func(saltandpepper_01)    
saltandpepper_01_erosion1=erosion_func(saltandpepper_01_dilation1)
saltandpepper_01_erosion2=erosion_func(saltandpepper_01_erosion1)
saltandpepper_01_dilation2=dilation_func(saltandpepper_01_erosion2) 
saltandpepper_01_dilation2_img=Image.fromarray(saltandpepper_01_dilation2)
saltandpepper_01_dilation2_img=saltandpepper_01_dilation2_img.convert("L")
saltandpepper_01_dilation2_img.save("saltandpepper_01_closing_opening_img.bmp")
print('saltandpepper_01_closing_opening_img:',snr_func(gray,saltandpepper_01_dilation2))














