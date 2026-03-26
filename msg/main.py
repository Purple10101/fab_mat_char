"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

main.py
Launcher — spawns ss3, ss4, ss5 as child processes and wires up their message queues.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import signal
import sys
import time
from multiprocessing import Process, Queue

from common import cprint, COLORS, RESET, BOLD
from ss3 import run_ss3
from ss4 import run_ss4
from ss5 import run_ss5


def main():
    print(f"""
{BOLD}{'=' * 58}
   SUBSYSTEM INTEGRATION
   ss3 (capture)  ->  ss4 (processing)  ->  ss5 (modelling)
{'=' * 58}{RESET}
""")

    # One inbox per subsystem
    ss3_q = Queue()
    ss4_q = Queue()
    ss5_q = Queue()

    # Each process gets its own inbox + a dict of the others' queues
    processes = [
        Process(
            target=run_ss3,
            name="ss3",
            daemon=True,
            args=(ss3_q, {"ss4": ss4_q, "ss5": ss5_q}),
        ),
        Process(
            target=run_ss4,
            name="ss4",
            daemon=True,
            args=(ss4_q, {"ss3": ss3_q, "ss5": ss5_q}),
        ),
        Process(
            target=run_ss5,
            name="ss5",
            daemon=True,
            args=(ss5_q, {"ss3": ss3_q, "ss4": ss4_q}),
        ),
    ]

    # Start ss3 and ss4 first so they're listening before ss5
    # kicks off the pipeline with its ready signal
    for p in processes:
        p.start()
        color = COLORS.get(p.name, "")
        cprint("system", f"Started {color}{p.name}{RESET}{COLORS['system']} (PID {p.pid})")
        time.sleep(0.5)

    print()
    cprint("system", "All subsystems running. Ctrl+C to stop.")
    cprint("system", "-" * 50)
    print()

    # Graceful shutdown
    def shutdown(sig, frame):
        print()
        cprint("system", "-" * 50)
        cprint("system", "Shutting down all subsystems...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                cprint("system", f"  Terminated {p.name} (PID {p.pid})")
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        cprint("system", "All subsystems stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor child processes
    try:
        while True:
            for p in processes:
                if not p.is_alive():
                    cprint("system", f"{p.name} died unexpectedly (exit code {p.exitcode})", "alert")
            time.sleep(2)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()