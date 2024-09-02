import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define min and max distances for normalization for each finger
distance_limits = {
    'thumb': (10, 130),   # (min_distance, max_distance)
    'index': (20, 110),
    'middle': (25, 120),
    'ring': (28, 110),
    'pinky': (30, 100)
}

def calculate_distance(point1, point2):
    # Calculate 3D Euclidean distance
    return np.linalg.norm(np.array(point1) - np.array(point2))

def normalize_distance(distance, min_distance, max_distance):
    # Normalize distance to a scale of 0 to 255
    normalized = (distance - min_distance) / (max_distance - min_distance)
    return int(np.clip(normalized * 255, 0, 255))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(frame_rgb)

    # List to store distances for each finger
    distances = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # List of (tip, mcp, finger_name) landmark pairs
            fingertip_mcp_pairs = [
                (4, 10, 'thumb'),   # Thumb
                (8, 5, 'index'),   # Index Finger
                (12, 9, 'middle'), # Middle Finger
                (16, 13, 'ring'),  # Ring Finger
                (20, 17, 'pinky')  # Pinky Finger
            ]

            for tip_idx, mcp_idx, finger_name in fingertip_mcp_pairs:
                # Get coordinates of the fingertip and MCP
                fingertip = hand_landmarks.landmark[tip_idx]
                mcp = hand_landmarks.landmark[mcp_idx]

                # Convert normalized coordinates to pixel values
                h, w, _ = frame.shape
                fingertip_coords = (int(fingertip.x * w), int(fingertip.y * h), fingertip.z)
                mcp_coords = (int(mcp.x * w), int(mcp.y * h), mcp.z)

                # Calculate distance
                distance = calculate_distance(fingertip_coords, mcp_coords)

                # Get the min and max distance for the current finger
                min_distance, max_distance = distance_limits[finger_name]

                # Normalize the distance
                normalized_distance = normalize_distance(distance, min_distance, max_distance)

                # Add the normalized distance to the list
                distances.append(normalized_distance)

                # Draw a line between the fingertip and MCP
                cv2.line(frame, (fingertip_coords[0], fingertip_coords[1]), 
                         (mcp_coords[0], mcp_coords[1]), (0, 255, 0), 2)

                # Display the normalized distance
                cv2.putText(frame, f'{normalized_distance}', (fingertip_coords[0], fingertip_coords[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            # Calculate bounding box for the palm
            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            x_min, x_max = min(x_coords) - 40, max(x_coords) + 40
            y_min, y_max = min(y_coords) - 40, max(y_coords) + 40

            # Check if all distances are 255
            if all(distance == 255 for distance in distances):
                # Draw a green rectangle with padding around the palm
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Print the distances to the terminal
    if distances:
        print("Normalized distances:", distances)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
