"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

ss3.py
image capture sub system emulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
import os
import pathlib
from multiprocessing import Queue
import cv2

from msg.src.common.common import Node, cprint, SharedImage
from msg.src.common.paths import IMAGE_PATH



class ImageCaptureSS3:
    """
    Simulates an image source with a stack of images and metadata.
    """
    def __init__(self):
        self.images = self._get_images()
        self.metadata = [
            {"image_id": 0, "x_mm": 100, "y_mm": 100, "valid": True},
            {"image_id": 1, "x_mm": 100, "y_mm": 100, "valid": True},
            {"image_id": 2, "x_mm": 100, "y_mm": 100, "valid": False},
            {"image_id": 3, "x_mm": 100, "y_mm": 100, "valid": True},
            {"image_id": 4, "x_mm": 100, "y_mm": 100, "valid": True},
            ]

    def _get_images(self):
        return [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith((".png", ".jpg", ".bmp"))]

    def pop(self):
        if self.images and self.metadata:
            filename = self.images.pop()
            meta = self.metadata.pop()
            return IMAGE_PATH / filename, meta
        return None, None

    @property
    def remaining(self):
        return len(self.images)


def run_ss3(inbox: Queue, peers: dict[str, Queue]):

    async def main():
        node = Node("ss3", inbox, peers)
        capture = ImageCaptureSS3()

        cprint("ss3", f"Images Ready. {capture.remaining} images in stack.")

        active_shm = None

        async def on_ready(msg):
            image_path, metadata = capture.pop()

            if image_path is not None:
                node.send("ss4", "image_data_message", {
                    "image_path": str(image_path),
                    "metadata": metadata,
                })
                cprint("ss3", f"Published {image_path.name} to ss4")
            else:
                node.send("ss4", "no_images", {"reason": "Stack empty"})

        node.on("ready_message", on_ready)

        while True:
            await node.poll()
            await asyncio.sleep(8)

    asyncio.run(main())
