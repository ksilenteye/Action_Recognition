import cv2
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model

# Use raw string or forward slashes for file path
model = load_model('video_classification_model.h5')

sys.stdout.reconfigure(encoding='utf-8')

# Define the action classes
classes = ["Basketball", "Diving", "HorseRace", "JumpRope", "VolleyballSpiking"]

# Frame extraction function
def frames_extraction(video_path, sequence_length=15):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize frame to (224, 224)
        frames_list.append(frame)
    video_reader.release()
    return frames_list

# Function to predict action in a single video
def predict_action(video_path, model, sequence_length=15):
    # Extract frames from the video
    frames = frames_extraction(video_path, sequence_length)
    
    # If there are fewer frames than the sequence length, skip the video
    if len(frames) < sequence_length:
        print("Not enough frames in video")
        return None
    
    # Convert frames to numpy array and normalize
    frames = np.array(frames).astype(np.float32)
    frames = (frames - frames.mean()) / frames.std()  # Normalize the frames
    
    # Reshape frames to match model input
    frames = frames.reshape(1, sequence_length, 224, 224, 3)  # Shape: (1, 15, 224, 224, 3)

    # Predict using the model
    predictions = model.predict(frames)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted index to class name
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Example usage
video_path = 'Video.mp4'  # Replace with your test video path
predicted_action = predict_action(video_path, model)

if predicted_action:
    print("Predicted Action:", predicted_action)
