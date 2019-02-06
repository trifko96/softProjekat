# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 
import collections
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.datasets import mnist



def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    ret,image_bin = cv2.threshold(image_gs, 73, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def dilate(image):
    kernel = np.ones((3,3))
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3))
    return cv2.erode(image, kernel, iterations=1)

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def scale_to_range(image): 
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in alphabet:
        output = np.zeros(10)
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
    
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32) 
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

def winner(output): 
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs):
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


"""(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_s = []
for x in x_train[:15000:]:
    ret, slika = cv2.threshold(x, 180, 255, cv2.THRESH_BINARY)
    x_train_s.append(slika)
    
ann = create_ann()
ann = train_ann(ann,np.array(prepare_for_ann(x_train_s),np.float32),convert_output(y_train[:15000:]))
model_json = ann.to_json()
with open("mreza.json", "w") as json_file:
    json_file.write(model_json)
    
ann.save_weights("mreza.h5")
print("Zavrseno obucavanje")"""



def pronadji_liniju(image, ann, lista1, lista2):
    img = image_gray(image)
    blur_img = cv2.GaussianBlur(img,(5, 5),0)
    edges = cv2.Canny(blur_img, 50, 150)
    """line_image = np.copy(image) * 0"""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, np.array([]),
                    50, 20)
    """cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),1)"""
    """lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)"""
    lista = []
    prva = []
    druga = []
    gornja = []
    donja = []
    suma = 0
    sx1 = 0
    sx2 = 0
    sy1 = 0
    sy2 = 0
    s1x1 = 0
    s1x2 = 0
    s1y1 = 0
    s1y2 = 0
    br = 0
    br1 = 0
    br2 = 0
    for line in lines:
        m = izracunaj_duzinu(line[0])
        if m > 200:
            lista.append(line)
            
    for line in lista:
        pom = izracunaj_duzinu(line[0])
        suma = suma + pom
        br += 1
        
    srVred = suma/br
    
    for line in lista:
        tmp = izracunaj_duzinu(line[0])
        if tmp > srVred:
            prva.append(line)
        else:
            druga.append(line)
            
    for line in prva:
        x1, y1, x2, y2 = line[0]
        sx1 += x1
        sy1 += y1
        sx2 += x2
        sy2 += y2
        br1 += 1
        
    svx1 = sx1/br1
    svy1 = sy1/br1
    svx2 = sx2/br1
    svy2 = sy2/br1
    
    for line in druga:
        x1, y1, x2, y2 = line[0]
        s1x1 += x1
        s1y1 += y1
        s1x2 += x2
        s1y2 += y2
        br2 += 1
        
    sv1x1 = s1x1/br2
    sv1y1 = s1y1/br2
    sv1x2 = s1x2/br2
    sv1y2 = s1y2/br2
    
    zbir1 = svy1 + svy2
    zbir2 = sv1y1 + sv1y2
        
    if zbir1 > zbir2:
        gornja.append(sv1x1)
        gornja.append(sv1y1)
        gornja.append(sv1x2)
        gornja.append(sv1y2)
        donja.append(svx1)
        donja.append(svy1)
        donja.append(svx2)
        donja.append(svy2)
    else:
        donja.append(sv1x1)
        donja.append(sv1y1)
        donja.append(sv1x2)
        donja.append(sv1y2)
        gornja.append(svx1)
        gornja.append(svy1)
        gornja.append(svx2)
        gornja.append(svy2)
        
    binarnaSlika = invert(image_bin(image_gray(image)))
    binarnaSlika1 = erode(dilate(binarnaSlika))
    slikag, regionig, lista1 = nadji_regione_ispod_gornje(image, binarnaSlika1, gornja, lista1)
    slikad, regionid, lista2 = nadji_regione_ispod_donje(image, binarnaSlika1, donja, lista2)
    #nadji_regione_ispod_gornje(image, binarnaSlika1, gornja)
    #nadji_regione_ispod_donje(image, binarnaSlika1, donja)

    zb = 0
    ra = 0
    if len(regionig) != 0:
        result = ann.predict(np.array(prepare_for_ann(regionig), np.float32))
        rezultat=[]
        rezultat = display_result(result)
        for r in rezultat:
            zb += r
    if len(regionid) != 0:
        result1 = ann.predict(np.array(prepare_for_ann(regionid), np.float32))
        rezultat1=[]
        rezultat1 = display_result(result1)
        for a in rezultat1:
            ra += a
    
    ukupno = zb - ra
    
    return lista1, lista2, ukupno
    
    
        
    
def nadji_regione_ispod_gornje(slika, binarna, linija, lista1):
    img, contours, hierarchy = cv2.findContours(binarna.copy(), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    mPrave, bPrave = izracunaj_koeficijente(linija)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        jedPr = mPrave*x + bPrave
        razlika = y - jedPr
        if h < 100 and h > 15 and w > 10 and x < (linija[2]-10) and (x+w) > (linija[0]+10) and y > jedPr and razlika <= 3:
            print(x,y,w,h)
            if uslov(x, y, w, h, lista1) == 0:    
                region = binarna[y:y+h+1,x:x+w+1]
                regions_array.append([resize_region(region), (x,y,w,h)]) 
                cv2.rectangle(slika,(x,y),(x+w,y+h),(0,255,0),2)
            lista1.append(x+y)
            lista1.append(w+h)
    if len(regions_array) != 0:
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = [region[0] for region in regions_array]
    return slika, sorted_regions, lista1

def nadji_regione_ispod_donje(slika, binarna, linija, lista2):
    img, contours, hierarchy = cv2.findContours(binarna.copy(), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    mPrave, bPrave = izracunaj_koeficijente(linija)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        jedPr = mPrave*x + bPrave
        razlika = y - jedPr
        if h < 100 and h > 15 and w > 10 and x < (linija[2]-10) and (x+w) > (linija[0]+10) and y > jedPr and razlika <= 3:
            print(x,y)
            if uslov(x, y, w, h, lista2) == 0:    
                region = binarna[y:y+h+1,x:x+w+1]
                regions_array.append([resize_region(region), (x,y,w,h)]) 
                cv2.rectangle(slika,(x,y),(x+w,y+h),(0,255,0),2)
            lista2.append(x+y)
            lista2.append(w+h)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return slika, sorted_regions, lista2
                
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def izracunaj_koeficijente(l):
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]
    m = (y2-y1)/(x2-x1)
    b = -m*x1 + y1
    return m,b
    
    
def izracunaj_duzinu(line):
    x1, y1, x2, y2 = line
    s = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)
    rez = s**(1.0/2)
    return rez

def uslov(x, y, w, h, lista):
    zbirKor = x + y
    zbirDim = w + h
    index = 0
    index1 = 0
    znak = 0
    znak1 = 0
    for el in lista:
        index += 1
        if zbirKor <= (el+2) and zbirKor >= (el-2):
            znak = 1
            break
    for el1 in lista:
        index1 += 1
        if zbirDim <= (el1+1) and zbirDim >= (el1-1):
            znak1 = 1
            break
    if znak == 1 and znak1 == 1 and abs(index1 - index) == 1:
        return 1
    else:
        return 0


pomFajl = open('mreza.json', 'r')
mrezaFajl = pomFajl.read()
pomFajl.close()
ann = model_from_json(mrezaFajl)
ann.load_weights("mreza.h5")
    
cap0 = cv2.VideoCapture('video/video-0.avi')
i0 = 0
suma0 = 0
lista1 = []
lista2 = []
while True:
    ret0, frame0 = cap0.read()
    if not ret0:
        break
    lista1, lista2, suma = pronadji_liniju(frame0, ann, lista1, lista2)
    suma0 += suma
    i0 += 1
    #plt.imshow(frame)
#print(suma0)
#print(i0)
cap0.release() 

cap1 = cv2.VideoCapture('video/video-1.avi')
i1 = 0
suma1 = 0
lista1 = []
lista2 = []
while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break
    lista1, lista2, suma = pronadji_liniju(frame1, ann, lista1, lista2)
    suma1 += suma
    i1 += 1
    #plt.imshow(frame)
#print(suma1)
#print(i1)
cap1.release() 

cap2 = cv2.VideoCapture('video/video-2.avi')
i2 = 0
suma2 = 0
lista1 = []
lista2 = []
while True:
    ret2, frame2 = cap2.read()
    if not ret2:
        break
    lista1, lista2, suma = pronadji_liniju(frame2, ann, lista1, lista2)
    suma2 += suma
    i2 += 1
    #plt.imshow(frame)
#print(suma2)
#print(i2)
cap2.release() 

cap3 = cv2.VideoCapture('video/video-3.avi')
i3 = 0
suma3 = 0
lista1 = []
lista2 = []
while True:
    ret3, frame3 = cap3.read()
    if not ret3:
        break
    lista1, lista2, suma = pronadji_liniju(frame3, ann, lista1, lista2)
    suma3 += suma
    i3 += 1
    #plt.imshow(frame)
#print(suma3)
#print(i3)
cap3.release() 

cap4 = cv2.VideoCapture('video/video-4.avi')
i4 = 0
suma4 = 0
lista1 = []
lista2 = []
while True:
    ret4, frame4 = cap4.read()
    if not ret4:
        break
    lista1, lista2, suma = pronadji_liniju(frame4, ann, lista1, lista2)
    suma4 += suma
    i4 += 1
    #plt.imshow(frame)
#print(suma4)
#print(i4)
cap4.release() 

cap5 = cv2.VideoCapture('video/video-5.avi')
i5 = 0
suma5 = 0
lista1 = []
lista2 = []
while True:
    ret5, frame5 = cap5.read()
    if not ret5:
        break
    lista1, lista2, suma = pronadji_liniju(frame5, ann, lista1, lista2)
    suma5 += suma
    i5 += 1
    #plt.imshow(frame)
#print(suma5)
#print(i5)
cap5.release() 

cap6 = cv2.VideoCapture('video/video-6.avi')
i6 = 0
suma6 = 0
lista1 = []
lista2 = []
while True:
    ret6, frame6 = cap6.read()
    if not ret6:
        break
    lista1, lista2, suma = pronadji_liniju(frame6, ann, lista1, lista2)
    suma6 += suma
    i6 += 1
    #plt.imshow(frame)
#print(suma6)
#print(i6)
cap6.release() 

cap7 = cv2.VideoCapture('video/video-7.avi')
i7 = 0
suma7 = 0
lista1 = []
lista2 = []
while True:
    ret7, frame7 = cap7.read()
    if not ret7:
        break
    lista1, lista2, suma = pronadji_liniju(frame7, ann, lista1, lista2)
    suma7 += suma
    i7 += 1
    #plt.imshow(frame)
#print(suma7)
#print(i7)
cap7.release() 

cap8 = cv2.VideoCapture('video/video-8.avi')
i8 = 0
suma8 = 0
lista1 = []
lista2 = []
while True:
    ret8, frame8 = cap8.read()
    if not ret8:
        break
    lista1, lista2, suma = pronadji_liniju(frame8, ann, lista1, lista2)
    suma8 += suma
    i8 += 1
    #plt.imshow(frame)
#print(suma8)
#print(i8)
cap8.release() 

cap9 = cv2.VideoCapture('video/video-9.avi')
i9 = 0
suma9 = 0
lista1 = []
lista2 = []
while True:
    ret9, frame9 = cap9.read()
    if not ret9:
        break
    lista1, lista2, suma = pronadji_liniju(frame9, ann, lista1, lista2)
    suma9 += suma
    i9 += 1
    #plt.imshow(frame)
#print(suma9)
#print(i9)
cap9.release() 

file = open("out.txt","w")
file.write("RA 79/2015 Pavle Trifkovic\r")
file.write("file	sum\r")
file.write('video-0.avi\t' + str(suma0) +'\r')
file.write('video-1.avi\t' + str(suma1) +'\r')
file.write('video-2.avi\t' + str(suma2) +'\r')
file.write('video-3.avi\t' + str(suma3) +'\r')
file.write('video-4.avi\t' + str(suma4) +'\r')
file.write('video-5.avi\t' + str(suma5) +'\r')
file.write('video-6.avi\t' + str(suma6) +'\r')
file.write('video-7.avi\t' + str(suma7) +'\r')
file.write('video-8.avi\t' + str(suma8) +'\r')
file.write('video-9.avi\t' + str(suma9) +'\r')
file.close()