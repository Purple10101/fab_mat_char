"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

ss4.py
image processing sub system emulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
from multiprocessing import Queue
import torch
import cv2
import os

from msg.src.common.common import Node, cprint, SharedImage
from msg.src.ss4.model import build_model
from msg.src.ss4.infer import run_inference


class ImageProcessingSS4:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device : {self.device}")

        self.model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=False)

        ckpt_path = os.path.join(os.path.dirname(__file__), "runs/fibre_maskrcnn/best.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded : {ckpt_path}")

    def run(self, image, metadata):
        """
        fabricated results for now
        """
        fibres = run_inference(self.model, image, self.device,
                               debug=True, debug_stem=f"frame_{metadata['image_id']}")
        ret_val = {
            "image_id": 99999,
            "char": [
                {"mesh_id": 0,
                 "dimensions": {
                     "length": 1.3,
                     "width": 0.06,
                 }},
                {"mesh_id": 1,
                 "dimensions": {
                     "length": 1.8,
                     "width": 0.03,
                 }},
                {"mesh_id": 2,
                 "dimensions": {
                     "length": 0.9,
                     "width": 0.03,
                 }},
                {"mesh_id": 3,
                 "dimensions": {
                     "length": 2.0,
                     "width": 0.08,
                 }},
            ]
        }
        return ret_val


def run_ss4(inbox: Queue, peers: dict[str, Queue]):

    async def main():
        node = Node("ss4", inbox, peers)
        proc = ImageProcessingSS4()

        cprint("ss4", f"Image Processor Ready.")

        async def on_publish_ready(msg):
            cprint("ss4", "Ready signal received — should push new data to client")

        async def on_image_data(msg):
            image_path = msg["data"]["image_path"]
            metadata = msg["data"]["metadata"]

            image = cv2.imread(image_path)

            cprint("ss4", f"Loaded {image_path} ({image.shape}) "
                          f"(x={metadata['x_mm']}, y={metadata['y_mm']})")

            result = proc.run(image, metadata)

            cprint("ss4", f"Processing complete: {result}")

            send_analysis(result)

        async def on_no_images(msg):
            cprint("ss4", "ss3 has no more images. Waiting.")

        def send_analysis(result):
            """Give results to ss5"""
            node.send("ss5", "processing_result", {
                "result": result
            })

        node.on("image_data_message", on_image_data)
        node.on("no_images", on_no_images) # this message wont be part of the real system
        node.on("ready_message", on_publish_ready)


        while True:
            await node.poll()
            await asyncio.sleep(8)

    asyncio.run(main())