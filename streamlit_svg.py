import streamlit as st
import cv2
from PIL import Image
import numpy as np
import keras.utils as image
from keras_vggface.utils import preprocess_input
import os
import joblib 
import keras_vggface
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

model = joblib.load('model_svr_resnet.h5')

def preprocessing(image_name):
    img = cv2.resize(image_name, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2)
    return img

vggface_resnet = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def predict_bmi():
    picKey = ' ' #Currently set to space

    #Set cascade file path and create the face cascade
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    #Set video source to default webcam
    video_capture = cv2.VideoCapture(0)

    #Initialize the count variable, used in naming frame images
    count = 0
    
    #Specify the font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    #Clears out any old images in this file
    for filename in os.listdir("./Webcam_Captures"):
        if filename.endswith(".jpg"):
            os.remove("./Webcam_Captures/" + filename)
    
    
    
    video_placeholder = st.empty()


    while True:
        #Capture frame-by-frame
        ret, frame = video_capture.read()

        #Detect faces
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor = 1.15,
            minNeighbors = 5,
            minSize = (30,30),
        )
        
        for (x, y, w, h) in faces:
            # Draw a blue rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            processed_img = preprocessing(frame[y:y+h, x:x+w])
            features = vggface_resnet.predict(processed_img)
            
            preds = model.predict(features)
            preds = round(preds[0],2)
            cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)
            #st.write(f'BMI: {preds}')
        #cv2.imshow('Predicting BMI', frame)
        video_placeholder.image(frame, channels="BGR")

        #Wait for keypress and save value
        capKey = cv2.waitKey(1)

        #Take a picture if picKey is pressed
        if capKey & 0xFF == ord(picKey):
            cv2.imwrite("./Webcam_Captures/frame%d.jpg" % count, frame)
            count += 1
            
        #Exit script if quitKey is pressed
        elif capKey & 0xFF == ord('q'):
            break
            
        
    video_capture.release()
    cv2.destroyAllWindows()

#def main():
    # Title and instructions
st.image("logo.png")
#video_tab, upload_tab = st.columns(2)
#st.title("BMI Prediction")
#st.write("Press the 'Predict BMI' button to start capturing and predicting BMI.")

video_tab, upload_tab = st.tabs(["Live Video", "Upload"])
with video_tab:
    st.write("Press the 'Predict BMI' button to start capturing and predicting BMI.")
    # if st.button("Predict BMI"):
    #     predict_bmi()
    if st.button("Predict BMI"):
        predict_bmi()

            
with upload_tab:
    file = st.file_uploader("Upload Art", key="file_uploader")
    if file is not None:
        try:
            img = Image.open(file)
        except:
            st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the file uploader, remove the image URL first.")

    #if st.button("Predict BMI"):
        #predict_bmi()

#if __name__ == "__main__":
    #main()