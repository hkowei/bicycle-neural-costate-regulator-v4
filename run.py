import subprocess
import sys
import ctypes
import winsound
import time

start_time = time.time()

try:
    print("========== Running bicycle training ==========")
    subprocess.run([sys.executable, "bi_train.py"], check=True)

    print("========== Running bicycle simulation ==========")
    subprocess.run([sys.executable, "bi_sim_ncr.py"], check=True)

    elapsed = time.time() - start_time
    msg = f"Training + simulation finished successfully.\nElapsed time: {elapsed/60:.1f} minutes"

    # print_config()

    winsound.Beep(1000, 800)
    ctypes.windll.user32.MessageBoxW(0, msg, "NCR Run Finished", 0x40)

except subprocess.CalledProcessError as e:
    elapsed = time.time() - start_time
    msg = f"Run failed.\nFile returned error code: {e.returncode}\nElapsed time: {elapsed/60:.1f} minutes"

    winsound.Beep(400, 1000)
    ctypes.windll.user32.MessageBoxW(0, msg, "NCR Run Failed", 0x10)