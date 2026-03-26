"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

ss5.py
modeling sub system emulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
from multiprocessing import Queue

from common import Node, cprint, SharedImage


class ModelingSS5:
    def __init__(self):
        print()

def run_ss5(inbox: Queue, peers: dict[str, Queue]):

    async def main():
        node = Node("ss5", inbox, peers)
        # kick off the first cycle
        node.send("ss3", "ready_message", {})
        cprint("ss5", "Sent ready signal to ss3 to kickstart pipeline")

        async def on_char_data(msg):
            result = msg["data"]["result"]
            image_id = result["image_id"]
            chars = result["char"]

            cprint("ss5", f"Received characterisation for image {image_id} "
                          f"({len(chars)} meshes)")

            for c in chars:
                mesh_id = c["mesh_id"]
                length = c["dimensions"]["length"]
                width = c["dimensions"]["width"]
                cprint("ss5", f"  mesh {mesh_id}: length={length}, width={width}")

            # Done processing — request the next one
            signal_ready()

        def signal_ready():
            node.send("ss3", "ready_message", {})
            node.send("ss4", "ready_message", {})
            cprint("ss5", "Sent ready signal to ss3 and ss4")

        node.on("processing_result", on_char_data)

        while True:
            await node.poll()
            await asyncio.sleep(10)

    asyncio.run(main())