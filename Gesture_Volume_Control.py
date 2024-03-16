import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
volume = 0.5 
vol_bar = 400
vol_per = 50
prev_volume = None  
prev_mute = False  

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_control = cast(interface, POINTER(IAudioEndpointVolume))


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def set_volume(distance):
    global volume
    global vol_per
    global vol_bar
    global prev_volume
    global prev_mute

    if prev_volume is None:
        prev_volume = volume

    prev_mute = volume_control.GetMute()

    if distance < 50:
        if volume > 0:
            volume -= 0.01

    elif distance > 100:
        if volume < 1:
            volume += 0.01

    vol_per = int(volume * 100)
    vol_bar = int(volume * 400)

    if volume != prev_volume or volume_control.GetMute() != prev_mute:
        try:
            if volume == 0:
                volume_control.SetMute(1, None)
            else:
                volume_control.SetMasterVolumeLevelScalar(volume, None)
        except Exception as e:
            print(f"Error setting volume: {e}")

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_finger = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                
                distance = calculate_distance((index_finger.x * frame.shape[1], index_finger.y * frame.shape[0]), 
                                              (thumb_finger.x * frame.shape[1], thumb_finger.y * frame.shape[0]))

                
                set_volume(distance)

        cv2.rectangle(frame, (10, 10), (10 + vol_bar, 30), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f'Volume: {vol_per}%', (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)


        cv2.imshow('Hand Gesture Volume Control', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
