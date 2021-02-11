# License Plate Recognition

## Drive Link with models:
https://drive.google.com/file/d/1gae-AXkgCj9liuEAHNmypdjOKrKkWJYN/view?usp=sharing <br /> put custom.weights file inside Data Folder and checkpoints/ folder in the main branch of the of the project (at the same level of data folder or core folder)

## Command to test with an image:
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/patente1.jpg --crop --info --plate

## Command to test with a video:
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/noticiero.mp4 --output ./detections/results.avi --plate
