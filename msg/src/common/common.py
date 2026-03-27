"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

common.py
system shared resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from datetime import datetime, timezone
from multiprocessing import Queue
from queue import Empty
import numpy as np
from multiprocessing import shared_memory


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Printing Res
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

COLORS = {
    "ss3": "\033[96m",   # Cyan
    "ss4": "\033[93m",   # Yellow
    "ss5": "\033[95m",   # Magenta
    "system":    "\033[92m",   # Green
    "alert":     "\033[91m",   # Red
    "dim":       "\033[90m",   # Grey
}
RESET = "\033[0m"
BOLD = "\033[1m"


def cprint(node_name: str, message: str, style: str = None):
    """Colour-coded, timestamped print."""
    color = COLORS.get(style or node_name, "")
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{COLORS['dim']}{ts}{RESET} {color}[{node_name}]{RESET} {color}{message}{RESET}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data and Routing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SharedImage:
    """
    helper to pass images between processes via shared memory.
    The image stoared in RAM once and only a small descriptor goes
    through the message broker.

    This is no good on windows but on linux this would work really good.
    """

    @staticmethod
    def publish(image: np.ndarray) -> tuple[dict, shared_memory.SharedMemory]:
        """
        Write an image into shared memory. Returns a descriptor
        dict and the shm handle. Caller MUST hold onto the handle
        until the receiver has read the data.
        """
        shm = shared_memory.SharedMemory(
            create=True,
            size=image.nbytes,
        )
        shared_array = np.ndarray(
            image.shape, dtype=image.dtype, buffer=shm.buf
        )
        np.copyto(shared_array, image)

        descriptor = {
            "shm_name": shm.name,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        }
        return descriptor, shm

    @staticmethod
    def receive(descriptor: dict) -> tuple[np.ndarray, shared_memory.SharedMemory]:
        """
        map to an existing shared memory block and return the image.
        Returns (image_copy, shm_handle).

        note: caller must call shm.close() and shm.unlink()
        when done, or use SharedImage.cleanup().
        """
        shm = shared_memory.SharedMemory(name=descriptor["shm_name"])
        shape = tuple(descriptor["shape"])
        dtype = np.dtype(descriptor["dtype"])

        shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Copy out of shared memory so the block can be freed immediately
        image = shared_array.copy()
        return image, shm

    @staticmethod
    def cleanup(shm: shared_memory.SharedMemory):
        """Close and free a shared memory block."""
        shm.close()
        shm.unlink()



class Node:
    """
    Lightweight message router for a single process.

    Args:
        name: This node's identifier.
        inbox: This node's incoming Queue.
        peers: Dict mapping peer names to their Queues.
    """

    def __init__(self, name: str, inbox: Queue, peers: dict[str, Queue]):
        self.name = name
        self.inbox = inbox
        self.peers = peers
        self._handlers: dict[str, callable] = {}

    def on(self, message_type: str, handler: callable):
        """Register an async handler for a message type."""
        self._handlers[message_type] = handler

    def send(self, target: str, message_type: str, data: dict):
        """Send a message to a specific peer."""
        if target not in self.peers:
            cprint(self.name, f"Unknown target: {target}", "alert")
            return

        msg = {
            "type": message_type,
            "sender": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self.peers[target].put(msg)
        cprint(self.name, f"──[ {message_type} ]──▶  {target}")

    def broadcast(self, message_type: str, data: dict):
        """Send a message to ALL peers."""
        targets = list(self.peers.keys())
        cprint(self.name, f"──[ {message_type} ]──▶  ALL ({', '.join(targets)})")
        for peer_name in self.peers:
            msg = {
                "type": message_type,
                "sender": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
            self.peers[peer_name].put(msg)

    async def poll(self, interval: float = 0.05):
        """
        Check the inbox for messages and dispatch them.
        Call this in your async loop — it's non-blocking.
        """
        try:
            while True:
                msg = self.inbox.get_nowait()
                msg_type = msg.get("type", "unknown")
                sender = msg.get("sender", "unknown")
                cprint(self.name, f"◀──[ {msg_type} ]──  {sender}")

                handler = self._handlers.get(msg_type)
                if handler:
                    await handler(msg)
                else:
                    cprint(self.name, f"No handler registered for '{msg_type}'", "alert")
        except Empty:
            pass  # Nothing in the queue right now
