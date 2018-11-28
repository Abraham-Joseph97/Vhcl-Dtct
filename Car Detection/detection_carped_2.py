





# -*- coding: utf-8 -*-

import cv2


cascade_src = 'cars.xml'

video_src = 'rc.3gpp'

cap = cv2.VideoCapture(video_src)
fgbg = cv2.createBackgroundSubtractorMOG2()
car_cascade = cv2.CascadeClassifier(cascade_src)
cap1 = cv2.VideoCapture(video_src)
fgbg1 = cv2.createBackgroundSubtractorMOG2()
bike_cascade = cv2.CascadeClassifier('pedestrian.xml')


while True:
    ret, img = cap.read()
    fgbg.apply(img)
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)


    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break
    ret, img = cap1.read()
	
    fgbg1.apply(img)
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bike = bike_cascade.detectMultiScale(gray,1.3,2)

    for(a,b,c,d) in bike:
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),4)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
