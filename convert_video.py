from moviepy.editor import VideoFileClip
from detect import Detection, create_windows
import numpy as np

from model import Model

if __name__ == "__main__":
    np.seterr(all='ignore')
    model = Model()
    windows = create_windows()
    detection = Detection(model, windows)

    clip1 = VideoFileClip("./data/videos/project_video.mp4") #.subclip(10, 40)
    # clip1 = VideoFileClip("./data/videos/test_video.mp4")
    white_clip = clip1.fl_image(detection.detect)
    white_clip.write_videofile("./output/project_video.mp4", audio=False)
    # white_clip.write_videofile("./output/test_video.mp4", audio=False)
