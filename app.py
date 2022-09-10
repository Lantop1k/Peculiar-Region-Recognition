import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template,session,Response
from flask import send_file
from werkzeug.utils import secure_filename
import cv2
import os
import dlib
import numpy as np
from facecaptured import getface
import time
import shutil
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.models import Sequential, load_model
from trainingfcn import train

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"]="filesystem"

app.secret_key = 'face_recog'

global img
user ='admin'
psw ='admin'
def gen_frames():
    while True:
        camera = cv2.VideoCapture(0)
        sucess, frame = camera.read()

        

        if not sucess:
            break
        else:
            cv2.imwrite('static/image/capture.png',frame)
            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +  frame + b'\r\n')


@app.route('/' ,methods=['POST','GET'])
def login():
    
    if request.method=='POST':
        username = request.form['username']
        password= request.form['password']

        if username == user and password==psw:
             return render_template('index.html',
                           uploadimg="../static/image/face.png",uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png")
        

    return render_template('login.html')

@app.route('/home' ,methods=['POST','GET'])
def index():
    
    if request.method=='POST':
        file = request.files['file']
        saveim = 'static/image/'+ secure_filename(file.filename)
        file.save(saveim)
        #imfile(file.filename)
        session['filen']=file.filename

        result,file2=getface(saveim)

        if result:
            
            predictor_path = 'shape_predictor_68_face_landmarks.dat'
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)

            img =  cv2.imread(file2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            w,h = gray.shape

            rectangle = dlib.rectangle(left=0, top=0, right=w, bottom=h)
            landmarks_dlib = predictor(gray, rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts

            landmark = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])

            x1,y1=landmark[17]
            x2,y2=landmark[21]
            x3,y3=landmark[41]


            left_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            left_file = 'static/image/left_eye'  + t + '.png'

            cv2.imwrite(left_file,left_eye)


            x1,y1=landmark[22]
            x2,y2=landmark[26]
            x3,y3=landmark[47]


            right_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            right_file = 'static/image/right_eye'  + t + '.png'

            cv2.imwrite(right_file,right_eye)

            img = image.load_img(right_file,target_size=(32,32))
            img = np.asarray(img)
            img = np.reshape(img,(1,32,32,3))

            json_file = open('weights_json.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            saved_model = model_from_json(loaded_model_json)
            saved_model.load_weights("weights.h5")

            #saved_model = load_model("weights.h5")

            output_idx = saved_model.predict(img)[0]

            persondb=os.listdir('database/RIGHT')

            person_detected = persondb[np.argmax(output_idx)]

            print('prediction idx :',output_idx) 

            

        #print('../'+saveim)       
        return render_template('index.html', uploadimg= '../'+file2,f_result='face detected',
                                   uploadimg2='../'+left_file,
                            uploadimg3='../'+right_file,detection=person_detected)
    
    return render_template('index.html',
                           uploadimg="../static/image/face.png",uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png")

@app.route('/video_feed' ,methods=['POST','GET'])
def video_feed():
    
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/capture' ,methods=['POST','GET'])
def capture():
    
    if request.method=='POST':

        file = '../static/image/capture.png'
        result,file2=getface(file[3:])

        if result:
            
            predictor_path = 'shape_predictor_68_face_landmarks.dat'
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)

            img =  cv2.imread(file2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            w,h = gray.shape

            rectangle = dlib.rectangle(left=0, top=0, right=w, bottom=h)
            landmarks_dlib = predictor(gray, rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts

            landmark = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])

            x1,y1=landmark[17]
            x2,y2=landmark[21]
            x3,y3=landmark[41]


            left_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            left_file = 'static/image/left_eye'  + t + '.png'

            cv2.imwrite(left_file,left_eye)


            x1,y1=landmark[22]
            x2,y2=landmark[26]
            x3,y3=landmark[47]


            right_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            right_file = 'static/image/right_eye'  + t + '.png'

            cv2.imwrite(right_file,right_eye)

            img = image.load_img(right_file,target_size=(32,32))
            img = np.asarray(img)
            img = np.reshape(img,(1,32,32,3))

            json_file = open('weights_json.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            saved_model = model_from_json(loaded_model_json)
            saved_model.load_weights("weights.h5")
            
            #saved_model = load_model("weights.h5")

            output_idx = saved_model.predict(img)[0]

            persondb=os.listdir('database/RIGHT')

            person_detected = persondb[np.argmax(output_idx)]

            return render_template('index.html', uploadimg= '../'+file2,f_result='face detected',
                                   uploadimg2='../'+left_file,
                            uploadimg3='../'+right_file,detection=person_detected)        


        else:
            return render_template('index.html', uploadimg="../static/image/face.png",uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png",
                                   f_result='No face detected')
        
        
    return render_template('index.html',
                           uploadimg="../static/image/face.png", uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png")

@app.route('/database' ,methods=['POST','GET'])
def database():
    
    if request.method=='POST':
        file = request.files['file']
        saveim = 'static/image/'+ secure_filename(file.filename)
        file.save(saveim)
        #imfile(file.filename)
        session['filen']=file.filename

        result,file2=getface(saveim)

        if result:
            
            predictor_path = 'shape_predictor_68_face_landmarks.dat'
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)

            img =  cv2.imread(file2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            w,h = gray.shape

            rectangle = dlib.rectangle(left=0, top=0, right=w, bottom=h)
            landmarks_dlib = predictor(gray, rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts

            landmark = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])

            x1,y1=landmark[17]
            x2,y2=landmark[21]
            x3,y3=landmark[41]


            left_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            left_file = 'static/image/left_eye'  + t + '.png'
            session['left_eye']=left_file
             

            cv2.imwrite(left_file,left_eye)

            x1,y1=landmark[22]
            x2,y2=landmark[26]
            x3,y3=landmark[47]


            right_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            right_file = 'static/image/right_eye'  + t + '.png'

            cv2.imwrite(right_file,right_eye)
            session['right_eye']=right_file

            #print('../'+saveim)       
            return render_template('database.html', uploadimg= '../'+file2,f_result='face detected',
                                       uploadimg2='../'+left_file,
                                uploadimg3='../'+right_file)
        else:

            return render_template('database.html', uploadimg="../static/image/face.png",uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png",
                                   f_result='No face detected')
          
    return render_template('database.html',
                           uploadimg="../static/image/face.png", uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png")


@app.route('/capturedb' ,methods=['POST','GET'])
def capturedb():
    
    if request.method=='POST':

        file = '../static/image/capture.png'
        result,file2=getface(file[3:])

        if result:
            
            predictor_path = 'shape_predictor_68_face_landmarks.dat'
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)

            img =  cv2.imread(file2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            w,h = gray.shape

            rectangle = dlib.rectangle(left=0, top=0, right=w, bottom=h)
            landmarks_dlib = predictor(gray, rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts

            landmark = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])

            x1,y1=landmark[17]
            x2,y2=landmark[21]
            x3,y3=landmark[41]


            left_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            left_file = 'static/image/left_eye'  + t + '.png'

            session['left_eye']=left_file

            cv2.imwrite(left_file,left_eye)


            x1,y1=landmark[22]
            x2,y2=landmark[26]
            x3,y3=landmark[47]


            right_eye = img[y1-20:y3+20,x1-20:x2+20]

            t=str(time.time())
            t=t[len(t)-5:]

            right_file = 'static/image/right_eye'  + t + '.png'

            session['right_eye']=right_file

            cv2.imwrite(right_file,right_eye)

            

            return render_template('database.html', uploadimg= '../'+file2,f_result='face detected',
                                   uploadimg2='../'+left_file,
                            uploadimg3='../'+right_file)        


        else:
            return render_template('database.html', uploadimg="../static/image/face.png",uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png",
                                   f_result='No face detected')
        
        
    return render_template('database.html',
                           uploadimg="../static/image/face.png", uploadimg2="../static/image/left_eye.png",
                            uploadimg3="../static/image/right_eye.png")

@app.route('/senddb' ,methods=['POST','GET'])
def senddb():
    if request.method=='POST':

        name=request.form['person_id']

        folder_left='database/LEFT/'+name
        folder_right='database/RIGHT/'+name

        if not(os.path.isdir(folder_left)):
            os.mkdir(folder_left)
            os.mkdir(folder_right)

        
        s_left = os.listdir(folder_left)
        s_right = os.listdir(folder_right)

        
        shutil.copyfile(session['left_eye'],folder_left + '/' + str(len(s_left))+'.jpg')
        shutil.copyfile(session['right_eye'],folder_right + '/' + str(len(s_right))+'.jpg')

        return render_template('database.html',
                           uploadimg="../static/image/face.png", uploadimg2=session['left_eye'],
                            uploadimg3=session['right_eye'],msg="Sucessfully Upload images to database")
        



@app.route('/training' ,methods=['POST','GET'])
def training():

    return render_template('report.html')



@app.route('/report' ,methods=['POST','GET'])
def report():

    hist = train()
    return render_template('report.html', trainaccuracy=round(hist.history["acc"][-1]*100),
                           valaccuracy=round(hist.history["val_acc"][-1]*100),
                           result='../static/image/training_result.png')






    

app.run(debug=True)
