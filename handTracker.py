import math
import time

import cv2
import mediapipe as mp
import pulsectl


class HandTracker:
    def __init__(self):
        self.pT = 0
        self.mHands = mp.solutions.hands
        self.handes = self.mHands.Hands()
        self.mDraw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.handes.process(img_rgb)

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                self.mDraw.draw_landmarks(frame, handlms, self.mHands.HAND_CONNECTIONS)

        return frame, self.result

    def get_fps(self):
        cT = time.time()
        fps = 1 / (cT - self.pT) if (cT - self.pT) != 0 else 0
        self.pT = cT
        return int(fps)

    def get_positions(self, img, hand_index=0):
        landmark_list = []
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[hand_index]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
        return landmark_list

    def distance(self, landmarks, index1, index2):
        id1, x1, y1 = landmarks[index1]
        id2, x2, y2 = landmarks[index2]

        distance = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        return distance


def main():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    tracker = HandTracker()

    while True:
        success, img = capture.read()

        if not success:
            print("Error: Failed to capture image.")
            break

        img, _ = tracker.process_frame(img)
        lmlist = tracker.get_positions(img)


        if len(lmlist) != 0:
            with pulsectl.Pulse('volume-control') as pulse:
                sink = pulse.sink_list()[0]  # default audio sink

                volume = min(tracker.distance(lmlist, 4,8) / 250, 1)  
                print(tracker.distance(lmlist, 4, 8))
                print(sink.volume.value_flat)
                pulse.volume_set_all_chans(sink,volume)



        fps = tracker.get_fps()

        cv2.putText(
            img, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3
        )
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
