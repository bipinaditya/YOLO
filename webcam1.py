import cv2 as cv
from random import randrange as r
traindata=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
webcam=cv.VideoCapture(0)
while True:
    success,img=webcam.read()
    grayimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    facecoordinates=traindata.detectMultiScale(grayimg)
    for x,y,w,h in facecoordinates:
        cv.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)
    cv.imshow('face',img)
    key=cv.waitKey(1)
    if(key==81 or key==113):
        break
webcam.release()
print("end of program")


