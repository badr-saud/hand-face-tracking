# importing computer vision library - opencv
import cv2
# hand tracking library
import mediapipe as mp
# calculating time
import time

pT = 0
cT = 0

# swap the webcam
capture = cv2.VideoCapture(0)  # internal webcam

# Check if the webcam is opened
if not capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

mHands = mp.solutions.hands
hands = mHands.Hands()
mDraw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()
    
    if not success:
        print("Error: Failed to capture image.")
        break

    imggrb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imggrb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mDraw.draw_landmarks(img, handlms, mHands.HAND_CONNECTIONS)

    cT = time.time()
    fps = 1 / (cT - pT) if (cT - pT) != 0 else 0
    pT = cT

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
