import os
import subprocess

# CURR = os.getcwd()
VID_DIR = "nlp_imp/keyframe_videos"

for filenname in os.listdir(VID_DIR):
    if filenname.endswith(".mp4"):
        input_pth = os.path.join(VID_DIR, filenname)
        temp_output_pth = os.path.join(VID_DIR, "temp_" + filenname)
        
        command = [
            "ffmpeg",
            "-i", input_pth,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-strict", "experimental",
            "-y",
            temp_output_pth
        ]
        
        print(f"fixing: {filenname}")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            os.replace(temp_output_pth, input_pth)
            print(f"[=] Fixed and replace: {filenname}")
        else:
            print(f"[!] Failed to fix: {filenname}")
            print(result.stderr.decode())

print("[=] COmplete COnversion.")