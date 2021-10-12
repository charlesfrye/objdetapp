import random
from datetime import datetime

import av
import cv2
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import torch
import wandb


model = torch.hub.load("ultralytics/yolov5", "yolov5s")

class LoggedObjDetProcessor(VideoProcessorBase):
    model = model

    def __init__(self):
        if wandb.run is None:
            wandb.init(project="objdetapp-sandbox", entity="charlesfrye", job_type="inference",
                       name=f"inference-streamlit-{get_now_strf()}")
        self.counter, self.max_log = 0, 32
        self.inference_table = wandb.Table(columns=["input", "output", "time"])
        self.table_logged = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = img[:, :, ::-1]  # convert bgr to rgb
        results = self.infer_img(img)
        processed_img = self.img_from_results(results)

        if (random.random() > 0.9) and wandb.run is not None:
            self.wandb_log(img, processed_img)

        return av.VideoFrame.from_ndarray(processed_img, format="rgb24")

    def infer_img(self, img):
        results = model(img)
        results.render()
        return results

    def wandb_log(self, img, processed_img):
        timestamp = get_now_strf()
        self.inference_table.add_data(wandb.Image(img), wandb.Image(processed_img), timestamp)
        self.counter += 1

        if self.counter >= self.max_log and wandb.run is not None:
            self.upload_table()
            wandb.finish()

    def img_from_results(self, results):
        img = results.imgs[0]
        return img

    def upload_table(self):
        wandb.log({"inference_results": self.inference_table})
        self.table_logged = True

    def on_ended(self):
        if wandb.run is not None:
            self.upload_table()
            wandb.finish()


def get_now_strf():
    now = datetime.now()
    timestamp = now.strftime("%Y/%m/%d %H:%M:%S")

    return timestamp


webrtc_streamer(key="objdet",
                video_processor_factory=LoggedObjDetProcessor)
