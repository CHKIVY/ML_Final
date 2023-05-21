import streamlit as st
import cv2
from PIL import Image
import numpy as np
import keras.utils as image
import os
import joblib 
import keras_vggface
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from streamlit_webrtc import webrtc_streamer
import av
import time,math,threading
from turn import get_ice_servers

model = joblib.load('model_svr_resnet.h5')
@st.cache_resource(show_spinner=False)
def preprocessing(image_name):
    img = cv2.resize(image_name, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2)
    return img

def run_model(processed_img):
    vggface_resnet = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    features = vggface_resnet.predict(processed_img)
    preds = model.predict(features)
    preds = round(preds[0],2)
    return preds


#lock = threading.Lock()
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX

###
thickness=10
#facesContainer = {'faces':[[0,0,0,0]]}


def video_frame_callback(frame):
    #img = frame.to_ndarray(format="bgr24")

    faces = faceCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), minNeighbors=2, 
                                        scaleFactor = 1.15,
                                        minSize = (30,30) )
    for (x,y,w,h) in faces:
            #x,y,w,h = expandBox((x,y,w,h),img.shape[1],img.shape[0])
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = frame[y:y+h, x:x+w]
        processed_img = preprocessing(image)
        preds = run_model(processed_img)
        st.write(preds)
        cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)
        #img = drawBoxNp(img,x,y,w,h,thickness)
    return preds, frame
    
class video_process:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None
        self.pred_bmi = []

    def recv(self, frame):
        frm = frame.to_ndarray(format='bgr24')
        pred_bmi, frame_bmi = video_frame_callback(frm)
        with self.frame_lock:
            self.out_image = frame_bmi
            self.pred_bmi = pred_bmi

        return av.VideoFrame.from_ndarray(frame_bmi, format="bgr24")

    #Set video source to default webcam
#     video_capture = cv2.VideoCapture(0)

#     #Initialize the count variable, used in naming frame images
#     count = 0
    
#     #Specify the font 
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     #Clears out any old images in this file
#     for filename in os.listdir("./Webcam_Captures"):
#         if filename.endswith(".jpg"):
#             os.remove("./Webcam_Captures/" + filename)
    
    
#     video_placeholder = st.empty()


#     while True:
#         #Capture frame-by-frame
#         ret, frame = video_capture.read()

#         #Detect faces
#         faces = faceCascade.detectMultiScale(
#             frame,
#             scaleFactor = 1.15,
#             minNeighbors = 5,
#             minSize = (30,30),
#         )
        
#         for (x, y, w, h) in faces:
#             # Draw a blue rectangle around the face
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#             processed_img = preprocessing(frame[y:y+h, x:x+w])
#             preds = run_model(processed_img)
            
#             cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)
#             #st.write(f'BMI: {preds}')
#         #cv2.imshow('Predicting BMI', frame)
#         video_placeholder.image(frame, channels="BGR")

#         #Wait for keypress and save value
#         capKey = cv2.waitKey(1)

#         #Take a picture if picKey is pressed
#         if capKey & 0xFF == ord(picKey):
#             cv2.imwrite("./Webcam_Captures/frame%d.jpg" % count, frame)
#             count += 1
            
#         #Exit script if quitKey is pressed
#         elif capKey & 0xFF == ord('q'):
#             break
            
        
#     video_capture.release()
#     cv2.destroyAllWindows()

# #def main():
#     # Title and instructions
st.image("logo.png")
# Sidebar
page_options = ['Business Applications', 'Model Demo']
selected_page = st.sidebar.selectbox("Select Page", page_options)

# Display icons and text based on the selected page
if selected_page == 'Business Applications':

    def resize_image(image_path, size):
        image = Image.open(image_path)
        image.thumbnail(size)
        return image

# Define the icon images and corresponding text
    icons = {
        'Icon 1': {
            'image_path': 'img1.png',
            'title': 'Social Health Research',
            'text': 'Gather public health data at large scale with surveillance cameras'
        },
        'Icon 2': {
            'image_path': 'img2.png',
            'title': 'Fitness App',
            'text': 'Quickly assess BMI without physical instrument'
        },
        'Icon 3': {
            'image_path': 'img3.png',
            'title': 'Virtual Try-ons',
            'text': 'Upload shopper images and virtually try on different outfits'
        }
    }

    # Set the size for the icon images
    image_size = (100, 100)

    # Page layout
    #st.subtitle("Business Applications")


    # Display three rows with icons and text
    row1, row2, row3 = st.columns(3)

    # Row 1
    with row1:
        st.image(resize_image(icons['Icon 1']['image_path'], image_size))
        st.caption(icons['Icon 1']['title'])
        st.write(icons['Icon 1']['text'])

    # Row 2
    with row2:
        st.image(resize_image(icons['Icon 2']['image_path'], image_size))
        st.caption(icons['Icon 2']['title'])
        st.write(icons['Icon 2']['text'])

    # Row 3
    with row3:
        st.image(resize_image(icons['Icon 3']['image_path'], image_size))
        st.caption(icons['Icon 3']['title'])
        st.write(icons['Icon 3']['text'])


elif selected_page == 'Model Demo':

    upload_tab, video_tab = st.tabs(["Upload", "Live Video"])

    with video_tab:
        st.write("Start predicting with the button below :arrow_down_small:")
        # if st.button("Predict BMI"):
        #     predict_bmi()
        #if st.button("Predict BMI"):
        
        ctx = webrtc_streamer(key="example", video_transformer_factory=video_process,
                sendback_audio=False, rtc_configuration={"iceServers": get_ice_servers()})

    with upload_tab:
        file = st.file_uploader("Upload Art", key="file_uploader")
        if file is not None:
            try:
                #img = Image.open(file)
                img = Image.open(file)
                st.image(img)
                image_name = np.array(img)
                # image_name = cv2.imread(img)
                processed_img = preprocessing(image_name)
                preds = run_model(processed_img)
                st.info(f"BMI: {preds}")

            except:
                st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    #if st.session_state.get("image_url") not in ["", None]:
    #    st.warning("To use the file uploader, remove the image URL first.")

    #if st.button("Predict BMI"):
        #predict_bmi()

# #if __name__ == "__main__":
#     #main()