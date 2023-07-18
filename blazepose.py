import cv2
import mediapipe as mp
import numpy as np

def calc_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Law of Cosine rule
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Replace 'path_to_your_video' with the actual path to your video
cap = cv2.VideoCapture('PushUpTest.mov')

# Desired width and height
width = 640
height = 960
counter = 0

with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:  # if frame read successfully
            # Resize the frame
            frame = cv2.resize(frame, (width, height))

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Joints
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate the angle
                angle = calc_angle(shoulder, elbow, wrist)

                # Display the angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                if angle > 170:
                    stage = "down"
                if angle < 50 and stage == "down":
                    stage = "up"
                    counter += 1
                    #print(counter)
            except:
                pass

            # Pushup display
            cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)

            cv2.putText(image, 'PUSH UPS', (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Pose estimation', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(len(landmarks))
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()