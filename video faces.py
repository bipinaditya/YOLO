import cv2 as cv
from random import randrange as r
traindata=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
vid=cv.VideoCapture('video.mp4')
while True:
    success,frame=vid.read()
    grayimg=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    facecoordinates=traindata.detectMultiScale(grayimg)
    for x,y,w,h in facecoordinates:
        cv.rectangle(frame,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)
    cv.imshow('face',frame)
    key=cv.waitKey(100)
    if(key==81 or key==113):
        break
frame.release()
print("end of program")