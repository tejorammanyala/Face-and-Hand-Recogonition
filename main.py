import cv2
import mediapipe as mp
import numpy as np

# Define colors for different facial parts
colors = [(128, 128, 128), (173, 216, 230), (144, 238, 144), (255, 255, 255)]

# Load the FaceMesh and HandTracking models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Function to detect faces and landmarks using MediaPipe
def detect_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces and landmarks in the image
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw colored facial parts
            draw_facial_parts(image, face_landmarks)

            # Draw outline for the overall face
            draw_face_outline(image, face_landmarks)

    # Detect hands and landmarks in the image
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and lines
            draw_hand_landmarks(image, hand_landmarks)

    return image

# Function to draw colored facial parts
def draw_facial_parts(image, face_landmarks):
    for i, landmark in enumerate(face_landmarks.landmark):
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

        # Determine the color index based on the landmark index
        color_index = i // 468  # There are 468 landmarks in total

        # Draw a circle for each landmark with the assigned color
        cv2.circle(image, (x, y), 1, colors[color_index], -1)

# Function to draw outline for the overall face
def draw_face_outline(image, face_landmarks):
    face_points = []

    # Extract the key points for the overall face outline
    for i in range(0, 17):
        landmark = face_landmarks.landmark[i]
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        face_points.append((x, y))

    # Convert the face points to a NumPy array
    face_points = np.array(face_points, dtype=np.int32)

    # Draw the outline for the overall face
    cv2.polylines(image, [face_points], True, (255, 255, 255), 2)

# Function to draw hand landmarks and lines
def draw_hand_landmarks(image, hand_landmarks):
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Draw lines connecting hand landmarks
    hand_connections = mp_hands.HAND_CONNECTIONS
    for connection in hand_connections:
        x0, y0 = int(hand_landmarks.landmark[connection[0]].x * image.shape[1]), int(hand_landmarks.landmark[connection[0]].y * image.shape[0])
        x1, y1 = int(hand_landmarks.landmark[connection[1]].x * image.shape[1]), int(hand_landmarks.landmark[connection[1]].y * image.shape[0])
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

# Load the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces, facial landmarks, and hand landmarks in the current frame
    frame = detect_landmarks(frame)

    # Display the frame
    cv2.imshow('3D Face and Hand Recognition', frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()