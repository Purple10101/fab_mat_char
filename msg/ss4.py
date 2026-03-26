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

from common import Node, cprint, SharedImage


def build_model(variant="maskrcnn_resnet50_fpn_v2", pretrained=True, trainable_backbone_layers=3):
    """
    Build a Mask R-CNN model fine-tuned for fibre segmentation.

    Args:
        variant                  : torchvision model name
        pretrained               : load COCO pre-trained weights
        trainable_backbone_layers: how many FPN layers to unfreeze (0-5)
                                   3 is a good default — keeps early layers frozen

    Returns:
        model: torch.nn.Module ready for training
    """
    import torchvision
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    NUM_CLASSES = 2  # 0 = background, 1 = fibre

    if variant == "maskrcnn_resnet50_fpn_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model   = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model   = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    # ── Replace box classifier head ──
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, NUM_CLASSES)

    # ── Replace mask head ──
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, NUM_CLASSES
    )

    return model



class ImageProcessingSS4:
    """
    Simulates an image processor.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device : {self.device}")
        #self.model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=False)

    def run(self, image, metadata):
        """
        fabricated results for now
        """
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