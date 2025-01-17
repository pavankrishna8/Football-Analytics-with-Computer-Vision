
from roboflow import Roboflow
rf = Roboflow(api_key="Enter your API key here")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

with open("dataset_location.txt", "w") as f:
    f.write(dataset.location)
