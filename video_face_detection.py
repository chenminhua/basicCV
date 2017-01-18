#!/usr/bin/env python
#coding=utf8
import numpy as np
import cv2

def detect(img, cascade):
    faces = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    # print faces   (x,y,w,h)
    if len(faces) == 0:
        return []
    faces[:,2:] += faces[:,:2]
    return faces

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# 分类器
cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rects = detect(gray, cascade)
    vis = img.copy()
    draw_rects(vis, rects, (0, 255, 0))

    cv2.imshow('facedetect', vis)

    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
