import torch
from transformers import pipeline
import pyaudio
import wave
import soundfile as sf
import threading
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pygame
from class_names import class_names
import string

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the Whisper pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    chunk_length_s=30,
    device=device,
)

# Define PyAudio parameters
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono channel
RATE = 16000  # 16kHz sampling rate
CHUNK = 1024  # Number of frames per buffer
WAVE_OUTPUT_FILENAME = "output.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Shared variable to control recording
recording = False

# Initialize pygame mixer
pygame.mixer.init()

# Load a sound file
sound = pygame.mixer.Sound("mixkit-urgent-simple-tone-loop-2976.wav")

# Load depth estimation model using pipeline
depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# Load YOLOv10 model
model = YOLO("yolov10x.pt")

# Open the webcam
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Cannot open the camera.")
    exit()

# Function to convert OpenCV frame to PIL image
def cv2_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Function to process a frame and generate a depth map using pipeline
def process_frame_for_depth(frame):
    pil_image = cv2_to_pil(frame)
    depth_output = depth_pipe(pil_image)
    depth_map = depth_output["depth"]
    depth_map = np.array(depth_map)

    # Invert depth values to ensure distance increases with actual distance
    max_depth = np.max(depth_map)  # Get maximum depth value
    depth_map = max_depth - depth_map  # Invert the depth map

    # Apply Gaussian blur to smooth the depth map
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

    return depth_map

# Function to record audio
def record_audio():
    global recording
    frames = []
    
    # Open stream for recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording... Press Enter to stop.")
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Recording finished")
    
    # Process the recorded audio
    process_audio(WAVE_OUTPUT_FILENAME)

# Function to process the audio
def process_audio(filename):
    global object_of_interest
    # Load the audio file
    audio_data, sampling_rate = sf.read(filename)

    # Prepare the sample in the expected format
    sample = {"array": audio_data, "sampling_rate": sampling_rate}

    # Get the transcription
    prediction = pipe(sample, batch_size=8)["text"]
    print("Transcription:", prediction)

    # Remove punctuation
    prediction = prediction.translate(str.maketrans("", "", string.punctuation))
    
    # Use alternative phrase "locate" or "show me"
    if "locate" in prediction.lower() or "find a" in prediction.lower() or "show me" in prediction.lower():
        words = prediction.lower().split()
        if "locate" in prediction.lower():
            index = words.index("locate")
        elif "find a" in prediction.lower():
            index = words.index("find") + 1  # Adjusted to "find a"
        elif "show me" in prediction.lower():
            index = words.index("show") + 1  # Adjusted to "show me"
        else:
            index = -1

        if index >= 0 and index + 1 < len(words):
            recognized_object = words[index + 1]
            if recognized_object in class_names.values():
                object_of_interest = recognized_object
                print(f"Object of interest set to: {object_of_interest}")
            else:
                print(f"{recognized_object} not recognized in the class names")
        else:
            print("No valid object of interest found in the transcription.")

# Main loop
def main():
    global recording
    while True:
        input("Press Enter to start recording...")
        recording = True
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()
        input("Press Enter to stop recording...")
        recording = False
        record_thread.join()

if __name__ == "__main__":
    # Initialize a flag to track if the object of interest is currently detected
    object_detected = False
    object_of_interest = None

    # Start the speech recognition in a separate thread
    threading.Thread(target=main, daemon=True).start()

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Detect objects using YOLOv10
        results = model(frame)[0]

        # Get bounding boxes, class labels, and class IDs
        bounding_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        class_labels = [class_names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        if object_of_interest:
            # Check if the object of interest is in the class labels
            if object_of_interest in class_names.values():
                object_of_interest_id = list(class_names.values()).index(object_of_interest)

                # Process the frame for depth estimation
                depth_map = process_frame_for_depth(frame)

                # Assume no object of interest detected initially
                current_frame_object_detected = False

                # Check if the object of interest is detected
                for bbox, class_id in zip(bounding_boxes, class_ids):
                    if class_id == object_of_interest_id:
                        current_frame_object_detected = True
                        x1, y1, x2, y2 = bbox
                        object_depth = np.mean(depth_map[y1:y2, x1:x2])  # Calculate average depth of the object

                        # Adjust sound volume based on object depth
                        volume = 1 - min(object_depth / np.max(depth_map), 1)  # Normalize and invert depth value for volume
                        sound.set_volume(volume)
                        break  # Stop searching once the object of interest is found

                if current_frame_object_detected:
                    if not pygame.mixer.get_busy():
                        sound.play(loops=-1)  # Play sound continuously
                else:
                    if pygame.mixer.get_busy():
                        sound.stop()

        # Display the annotated frame
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release the camera and close windows
    capture.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()  # Quit pygame mixer
    audio.terminate()
