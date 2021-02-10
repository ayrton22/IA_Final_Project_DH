# License Plate Recognition

## Command to test with an image:
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/patente1.jpg --crop --info --plate

## Command to test with a video:
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/noticiero.mp4 --output ./detections/results.avi --plate
