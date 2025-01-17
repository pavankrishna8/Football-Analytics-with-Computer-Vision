import subprocess

# Read the dataset location from the file
with open("dataset_location.txt", "r") as f:
    dataset_location = f.read().strip()

# Train the YOLOv5 model using subprocess to run the shell command
command = [
    "yolo",
    "task=detect",
    "mode=train",
    f"model=yolov5x.pt",
    f"data={dataset_location}/data.yaml",
    "epochs=100",
    "imgsz=640"
]

subprocess.run(command, shell=True)
