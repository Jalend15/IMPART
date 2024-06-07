import cv2
import os
import torch
from torchvision import models, transforms
from PIL import Image

output_dir = "2020_embeddings/"

def extract_frames(video_path, file,fps=4):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the FPS of the video
    print(video_fps)
    
    frame_count = 0
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frames at the specified FPS

        if frame_count % (video_fps // fps) == 0:
            frame_no+=1
            frame_path = os.path.join(output_dir, f"{file}_frame_{frame_no}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_path}")
        
        frame_count += 1
        
    print(frame_count)
    print(frame_no)
    print(video_fps // fps)

    cap.release()
    
# Define the directory to scan
directory_path = '2020/'

# Specify the file extension of files to delete
necessary_extension = '.mp4'

# Walk through the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        print(directory_path+file)
        extract_frames(directory_path+file,file)
        break
    break


# # Load a pre-trained model (e.g., ResNet)
# model = models.resnet50(pretrained=True)
# model.eval()

# # Define the transformation
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def generate_embeddings(frame_path):
#     image = Image.open(frame_path)
#     tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
#     with torch.no_grad():
#         embeddings = model(tensor)
    
#     return embeddings.numpy()

# # Example usage for one frame
# embeddings = generate_embeddings('/path/to/frame.jpg')
# print(embeddings)
