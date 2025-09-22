import os
import subprocess
files = os.listdir("video")
for file in files:
    #print(file)
    tutorial_no = file.split(". ")[0]
    name = file.split(". ")[1].split(".")[0]
    #print(tutorial_no , name)

    subprocess.run(["ffmpeg" , "-i" , f"video/{file}" , f"audios/{tutorial_no}_{name}.mp3"])