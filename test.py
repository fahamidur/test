import cv2
from glob import glob
from pathlib import Path

from model import Model
from utils import plot

def process_video(input_video_path: str, output_path: str):
    # video_path
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print('FPS', fps)
    success, frame = cap.read()
    
    out = cv2.VideoWriter(save_video_path, fourcc, fps, size)
    success = True


    while success and cap.isOpened():

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        print(f"label: {label} confidence: {conf}")

        # cv2.imshow('Recording...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
        success, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()


model = Model()

image_pathes = glob('./data/*.jpg')
image_pathes += glob('./data/*.png')
image_pathes[:2]
save_folder_path = Path('results')

video_path = './data/fire.mp4'
save_video_path = str(save_folder_path / 'result_fire.avi')

process_video(video_path, save_video_path)
