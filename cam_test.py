import cv2
import os
import time


camera_url = "rtmp://10.10.10.130:1935/live/stream"


# Start the webcam
cap = cv2.VideoCapture(camera_url)

end_time = time.time() + 15

while time.time() < end_time:
        ret, frame = cap.read()
        cv2.imshow(f"Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
