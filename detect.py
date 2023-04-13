import cv2 as cv
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


mask_model = load_model('model/kaggle/working/mask_model')

f_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


class MaskDetector():
    
    def __init__(self,):
         pass
    
    def detect_mask(self):
        
#         img_file = path_to_img_file

        cam_capture = cv.VideoCapture(0)
        
        while True:
            ret, frame = cam_capture.read()

            faces = MaskDetector.detect_faces(frame)

            for face in faces:
                x,y,w,h = face
                roi = frame[y:y+h, x:x+w]
#                 cv.imshow('ROI', roi)
                mask_prob = MaskDetector.predict_image_with_mask(roi, mask_model) 
                frame = MaskDetector.draw_bounding_box(frame, face, mask_prob)
        
            cv.imshow('Face', frame)
            
            if cv.waitKey(1) & 0xff == ord('q'):
                break
                
        cam_capture.release()
        cv.destroyAllWindows()
            
    def preprocess_image(img):
        img_arr = img_to_array(img)
        img_arr = cv.resize(img_arr, (150, 150))
        img_arr = np.expand_dims(img_arr, axis=0)
        return img_arr/255.

    def predict_image_with_mask(img, model):
        img_arr = MaskDetector.preprocess_image(img)
        probability = 1 - model.predict(img_arr).flatten()[0]
        print(f"probablity of mask present: {probability:.04}")
        return probability
    
    def detect_faces(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = f_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
        return faces
    
    def draw_bounding_box(frame, face, mask_prob):
        color = None
        text = None
        disp_prob = 0

        (x,y,w,h) = face
        if mask_prob >= 0.5:
            color = (0,255,0)
            text = 'Mask'
            disp_prob = mask_prob
        else:
            color = (0,0,255)
            text = 'No Mask'
            disp_prob = 1 - mask_prob

        cv.rectangle(frame, (x,y), (x+w, y+h), color, thickness=1)
        cv.putText(frame, f'{text} {round(disp_prob*100, 2)}%', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=color, thickness=1, lineType=cv.LINE_AA, bottomLeftOrigin=False)
        return frame
     
        
if __name__ == '__main__':
    mask_detector = MaskDetector()
    mask_detector.detect_mask()