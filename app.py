from datetime import datetime
from pathlib import Path
import random

import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import torch
import wandb

artifacts_dir = Path("artifacts")
ENTITY = "charlesfrye"
PROJECT = "objdetapp"
MODEL = "run_2r7xmnog_model"

logged_data = ["confidence", "class", "name"]


class LoggedObjDetProcessor(VideoProcessorBase):

    def __init__(self):
        if wandb.run is None:
            wandb.init(project="objdetapp",
                       entity="charlesfrye",
                       job_type="inference",
                       name=f"inference-streamlit-{get_now_strf()}")
        self.counter, self.max_log = 0, 256
        columns = ["input", "output", "time"] + logged_data
        self.inference_table = wandb.Table(columns=columns)
        self.table_logged = False

        weights_path = self.setup_weights(ENTITY, PROJECT, MODEL)

        self.model = torch.hub.load("ultralytics/yolov5", "custom", weights_path)

    def recv(self, frame):
        img = frame.to_image()
        results = self.infer_img(img)
        processed_img = self.img_from_results(results)

        if (random.random() > 0.9) and wandb.run is not None:
            self.wandb_log(img, processed_img, results)

        return av.VideoFrame.from_ndarray(processed_img, format="rgb24")

    def infer_img(self, img):
        results = self.model(img)
        results.render()
        return results

    def wandb_log(self, img, processed_img, results):
        timestamp = get_now_strf()
        results_contents = self.get_logged_data(results)
        self.inference_table.add_data(
            wandb.Image(img), wandb.Image(processed_img), timestamp, *results_contents)
        self.counter += 1

        if self.counter >= self.max_log and wandb.run is not None:
            self.upload_table()
            wandb.finish()

    def setup_weights(self, entity, project, model):
        version = get_best_version(f"{entity}/{project}/{model}")
        artifact_name = f"{model}:{version}"
        artifact_dir = artifacts_dir / artifact_name
        model_artifact = f"{entity}/{project}/{artifact_name}"
        model_artifact = wandb.run.use_artifact(model_artifact)
        artifact_dir = model_artifact.download()

        weights_path = Path(artifact_dir) / "best.pt"

        return weights_path

    def img_from_results(self, results):
        img = results.imgs[0]
        return img

    def get_logged_data(self, results):
        results = results.pandas()
        result = results.xyxy[0]
        if len(result) == 0:
            return [None] * len(logged_data)
        top_box = result.sort_values(by="confidence", ascending=False).iloc[0]
        return top_box[logged_data]

    def upload_table(self):
        wandb.log({"inference_results": self.inference_table})
        self.table_logged = True

    def on_ended(self):
        if wandb.run is not None:
            self.upload_table()
            wandb.finish()


def get_best_version(model_artifact):
    api = wandb.Api()
    for artifact_version in api.artifact_versions("model", model_artifact):
        if "best" in artifact_version.aliases:
            return artifact_version.version
    raise ValueError(f"artifact {model_artifact} has no version labeled 'best'")


def get_now_strf():
    now = datetime.now()
    timestamp = now.strftime("%Y/%m/%d %H:%M:%S")

    return timestamp


webrtc_streamer(key="objdet",
                video_processor_factory=LoggedObjDetProcessor)
