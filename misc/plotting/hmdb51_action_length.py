import torchvision as tv
from glob import glob
from os.path import join, exists
from tqdm import tqdm
import numpy as np

DATA_ROOT = "/ssd/fmthoker/hmdb51/videos"


if __name__ == "__main__":

    video_paths = glob(join(DATA_ROOT, "*", "*.avi"))
    
    
    iterator = tqdm(video_paths, desc="Processing")
    
    durations = []
    for video_path in iterator:
        # video = tv.io.VideoReader(video_path, "video")
        timestamps, fps = tv.io.read_video_timestamps(video_path, pts_unit="sec")
        duration = len(timestamps) / fps
        durations.append(duration)
    
    print("Mean video length: {}".format(np.mean(durations)))

