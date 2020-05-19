import cv2,dlib
import numpy as np

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
num=0
def find_faces(img):
    dets = detector(img,1)
    
    if len(dets) == 0:
        print(0)
    else:
        print(len(dets))
    
    for k,d in enumerate(dets):
        left = d.left()
        right = d.right()
        top = d.top()
        bottom = d.bottom()

        crop_image = img[top:bottom,left:right]
        cv2.imwrite('YoungHwan/YH_'+str(num)+'.jpg',crop_image)


while num<100:
    ret,img_bgr = cap.read()
    if not ret:
        break

    find_faces(img_bgr)
    num+=1
    cv2.imshow('',img_bgr)
    if cv2.waitKey(1)>0:break

cap.release()
cv2.destroyAllWindows()
