import cv2,dlib
import numpy as np
import sys,os

cap = cv2.VideoCapture('sobin_sample.mov')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
num=0
name = input('이름을 입력하세요 자동으로 촬영이 시작됩니다. : ')
last_found = {'name': 'unknown', 'dist': 0.5, 'color': (0,0,255)}



def find_faces(img):
    dets = detector(img,1)

    if len(dets)==0:
        print(0)
    else:
        print(len(dets))

    for k,d in enumerate(dets):
        left = d.left()
        right = d.right()
        top = d.top()
        bottom = d.bottom()

        cv2.rectangle(img, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
        
        crop_image = img[top:bottom,left:right]
        cv2.imwrite(f'./UserData/{name}/{name}'+str(num)+'.jpg',crop_image)
        

try:
    if not(os.path.isdir(f'./UserData/{name}')):
        os.makedirs(os.path.join(f'./UserData/{name}'))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise
num=200
while num>-1:
    ret,img_bgr = cap.read()
    height, width, channel = img_bgr.shape
    img_bgr = cv2.resize(img_bgr,(width//4,height//4))
    if not ret:
        break
    
    matrix = cv2.getRotationMatrix2D((width/2, height/2), -90, 1)
    img_bgr = cv2.warpAffine(img_bgr, matrix, (width, height))
    
    find_faces(img_bgr)
    cv2.imshow('img',img_bgr)
    num+=1
    if cv2.waitKey(1)>0:break
    
cap.release()
cv2.destroyAllWindows()
