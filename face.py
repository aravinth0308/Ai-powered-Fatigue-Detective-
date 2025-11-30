import cv2
import dlib
import simpleaudio as sa
from scipy.spatial import distance

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize dlib’s face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmarks indexes
(lStart, lEnd) = (42, 48)  # Left eye
(rStart, rEnd) = (36, 42)  # Right eye

# Thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Variables
counter = 0
alarm_playing = False
play_obj = None

# Load alarm
alarm_path = "D:/jai/GIT/Projects/Driver_Fatique_Monitoring_and_Alert_System/drowsiness_alarm.wav"
wave_obj = sa.WaveObject.from_wave_file(alarm_path)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw EAR on screen
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= EAR_CONSEC_FRAMES:
                if not alarm_playing:
                    print("[ALERT] Drowsiness detected!")
                    play_obj = wave_obj.play()
                    alarm_playing = True
        else:
            counter = 0

    # Show the frame
    cv2.imshow("Driver Monitor", frame)

    # Key press controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and alarm_playing:
        # Stop the alarm
        if play_obj is not None:
            play_obj.stop()
        alarm_playing = False
        print("[INFO] Alarm manually stopped by driver.")

    elif key == ord('q'):
        # Quit the program
        if alarm_playing and play_obj is not None:
            play_obj.stop()
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
