# importing computer vision library - opencv
import cv2
# hand tracking library
import mediapipe as mp
# calculating 
import time

pT = 0
cT = 0


# swap the webcam
capture = cv2.VideoCapture(0) # internal webcam
mHands = mp.solutions.hands
hands = mHands.Hands()
mDraw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()

    imggrb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imggrb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mDraw.draw_landmarks(img,handlms, mHands.HAND_CONNECTIONS)

    cT = time.time()
    fps = 1 / (cT-pT)
    pT = cT

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow("Hand tracking", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()