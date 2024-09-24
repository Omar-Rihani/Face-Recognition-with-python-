import cv2
import numpy as np
import face_recognition
import os

# Path to the folder containing images for face recognition
img_directory = 'IMAGES'
img_list = []
person_names = []

# List of all persons' images from the directory
person_files = os.listdir(img_directory)

# Loading each image and extracting the person name from the filename
for person in person_files:
    img_file = cv2.imread(f'{img_directory}/{person}')
    img_list.append(img_file)
    person_names.append(os.path.splitext(person)[0])
print(person_names)

# Function to encode the loaded images
def encode_images(img_list):
    encoding_list = []
    for img in img_list:
        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get the face encoding for the image
        img_encoding = face_recognition.face_encodings(img_rgb)[0]
        encoding_list.append(img_encoding)
    return encoding_list

# Encode all the images and store the encodings
known_encodings = encode_images(img_list)
print('Encoding Complete.')

# Start capturing video from the webcam 
video_capture = cv2.VideoCapture(1)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()
    
    # Resize frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    current_face_locations = face_recognition.face_locations(small_frame_rgb)
    current_face_encodings = face_recognition.face_encodings(small_frame_rgb, current_face_locations)

    # Loop through all detected faces
    for face_encoding, face_location in zip(current_face_encodings, current_face_locations):
        # Compare the detected face encodings with the known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # Find the best match (smallest distance)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            detected_name = person_names[best_match_index].upper()
            print(detected_name)

            # Scaling the face location back to the original frame size
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Drawing a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, detected_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Wait for a key press and break if 'y' is pressed
    if cv2.waitKey(1) & 0xFF == ord('y'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
