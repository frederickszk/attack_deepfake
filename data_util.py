import numpy as np
import os
from os.path import join
import PIL.Image as img

class VideoReader:
    def __init__(self):
        self.index = []
        self.frame_num = 5
        self.data_root = ""

        self.capacity = 0
        # self.count = 0

    def initialize(self, data_root, frame_num=5):
        index = []
        # For fake samples
        for video in os.listdir(join(data_root, "fake")):
            sample = []
            count = 0
            for frame in os.listdir(join(data_root, "fake", video)):
                sample.append(join(data_root, "fake", video, frame))
                count += 1
                if count % frame_num == 0 and count != 0:
                    index.append([sample, 0])
                    sample = []

        # For real samples
        for video in os.listdir(join(data_root, "real")):
            sample = []
            count = 0
            for frame in os.listdir(join(data_root, "real", video)):
                sample.append(join(data_root, "real", video, frame))
                count += 1
                if count % frame_num == 0 and count != 0:
                    index.append([sample, 1])
                    sample = []

        self.index = np.array(index)
        self.capacity = self.index.shape[0]

    def generate(self, img_size=(300, 300), batch_size=5):
        
        # While Loop for generate
        count = 0
        np.random.shuffle(self.index)
        while count < self.capacity:
            video_batch = []
            label_batch = []
            for i in range(batch_size):
                video = []
                for frame_address in self.index[count][0]:
                    video.append(np.array(img.open(frame_address).resize(img_size))/255)
                video_batch.append(np.array(video))
                label_batch.append(self.index[count][1])
                count += 1
                if count >= self.capacity:
                    break
            yield np.array(video_batch), np.array(label_batch)