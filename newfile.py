import cv2
import mediapipe as mp
import pyttsx3
import time
import numpy as np

engine = pyttsx3.init()
engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

GESTURES = {
    0: ("Hello", "Open hand"),
    1: ("OK/Yes", "Thumbs up"),
    2: ("No", "Thumbs down"), 
    3: ("You", "Pointing finger")
}

class GestureRecognizer:
    def __init__(self):
        self.last_spoken = ""
        self.last_time = 0
        self.cooldown = 2

    def is_finger_extended(self, fingertip, pip, dip):
        return fingertip.y < pip.y and fingertip.y < dip.y

    def recognize(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return None

        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        
        index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_dip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_dip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_dip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
        pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        pinky_dip = landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
        thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

        index_extended = self.is_finger_extended(index_tip, index_pip, index_dip)
        middle_extended = self.is_finger_extended(middle_tip, middle_pip, middle_dip)
        ring_extended = self.is_finger_extended(ring_tip, ring_pip, ring_dip)
        pinky_extended = self.is_finger_extended(pinky_tip, pinky_pip, pinky_dip)
        
        thumb_extended = thumb_tip.y < thumb_ip.y

        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )

        if (index_extended and not middle_extended and 
            not ring_extended and not pinky_extended and
            thumb_index_dist > 0.15):
            return 3
            
        elif (thumb_extended and not index_extended and 
              not middle_extended and not ring_extended and not pinky_extended):
            return 1
            
        elif (not thumb_extended and not index_extended and 
              not middle_extended and not ring_extended and not pinky_extended):
            return 2
            
        elif all([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
            return 0
            
        return None

    def speak_gesture(self, gesture_id):
        current_time = time.time()
        if current_time - self.last_time < self.cooldown:
            return
            
        gesture_text = GESTURES.get(gesture_id, ("Unknown", ""))[0]
        
        if gesture_text != self.last_spoken:
            print(f"Detected gesture: {gesture_text}")
            engine.say(gesture_text)
            engine.runAndWait()
            self.last_spoken = gesture_text
            self.last_time = current_time

def main():
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)
    
    window_name = 'Gesture Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        current_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
                
                current_gesture = recognizer.recognize(hand_landmarks, w, h)
                
                if current_gesture is not None:
                    recognizer.speak_gesture(current_gesture)
                    gesture_text, _ = GESTURES.get(current_gesture, ("Unknown", ""))
                    cv2.putText(frame, gesture_text, (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()