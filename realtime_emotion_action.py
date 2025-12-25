import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model


emotion_model = load_model("emotion_model.h5", compile=False)
action_model  = load_model("action_model.h5", compile=False)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
action_labels  = ["BrushingTeeth", "Typing", "Punch", "WritingOnBoard"]
SEQUENCE_LENGTH = 15
ACTION_FRAME_SIZE = (128, 128)

FACE_INTERVAL    = 5
EMOTION_INTERVAL = 5
ACTION_INTERVAL  = 5

TARGET_FPS = 12                    
FRAME_DELAY = 1.0 / TARGET_FPS

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

#Cam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.perf_counter()
frame_count = 0

faces = []
emotion_text = "No Face"
action_text  = "No Action"


# While Loop

while True:
    loop_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

#Detection
    if frame_count % FACE_INTERVAL == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

    
    # IF FACE EXISTS
    
    if len(faces) > 0:
        x, y, fw, fh = faces[0]

        # EMOTION 
        if frame_count % EMOTION_INTERVAL == 0:
            face = gray[y:y+fh, x:x+fw]
            face = cv2.resize(face, (48, 48))
            face = face.astype(np.float32) / 255.0
            face = face.reshape(1, 48, 48, 1)

            pred = emotion_model.predict(face, verbose=0)
            idx = np.argmax(pred)
            emotion_text = f"{emotion_labels[idx]} ({pred[0][idx]*100:.1f}%)"

        # ACTION 
        resized = cv2.resize(frame, ACTION_FRAME_SIZE)
        resized = resized.astype(np.float32) / 255.0
        frame_buffer.append(resized)

        if len(frame_buffer) < SEQUENCE_LENGTH:
            action_text = f"Collecting ({len(frame_buffer)}/{SEQUENCE_LENGTH})"
        elif frame_count % ACTION_INTERVAL == 0:
            seq = np.expand_dims(frame_buffer, axis=0)
            pred = action_model.predict(seq, verbose=0)
            action_text = action_labels[np.argmax(pred)]

        cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,255,0), 2)

    # IF NO FACE
    else:
        emotion_text = "No Face"
        action_text  = "No Action"
        frame_buffer.clear()   


    # FPS CALCULATION
    curr_time = time.perf_counter()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # DISPLAY

    cv2.putText(frame, f"Emotion: {emotion_text}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"Action: {action_text}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Emotion & Action ", frame)

    # FPS LIMITER

    elapsed = time.perf_counter() - loop_start
    if elapsed < FRAME_DELAY:
        time.sleep(FRAME_DELAY - elapsed)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
