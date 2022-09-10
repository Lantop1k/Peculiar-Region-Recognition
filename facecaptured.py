import cv2
import time


# haarcascade cascade classifier file for face extraction
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)



def getface(file):

    img=cv2.imread(file) # reading image file   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # converting the color of the image to gray color
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # detecting faces from the image using detectmultiscale

    print('faces',faces)
    print(len(faces))
    if len(faces)==1:
        x, y, w, h = faces[0]

        cropface = img[y:y+h,x:x+w]

        t=str(time.time())
        t=t[len(t)-5:]

        file2 = 'static/image/crop_face'+t+'.png'
        cv2.imwrite(file2,cropface)

        return True,file2

    else:

        return False,file

        
            
        
