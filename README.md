# Custom Detectron 2 
## For the AirDetection Master Thesis

This Repository has been used to produce the Faster R-CNN models used in the remote sensing obejct detection. 
It is a part of the AirDetection study. The Main repo can be found [under this link](https://github.com/theATM/AirDetection).

[Original Detectron2 README](repo/README.md)

The models are trained on the RSD-GOD dataset.

## Usage

Run main.py with the python enviroment of choice (DockerFile avaiable)

Use three functions train() validate() and predict() to train test and detect on RSD-GOD dataset 

To detect on custom images run the repo/demo/demo.py script with those params:
* config-file  ../configs/COCO-Detection/faster_rcnn_custom.yaml
* input image_dir
* output save_dir
* opts MODEL.WEIGHTS 'model.pth'


## Results

![Untitled](https://github.com/theATM/Detectron2/assets/48883111/76fa0c30-6fd9-44a4-a7ff-a9247f028e55)

Pretrained R20 Model [Here](https://drive.google.com/file/d/1ICOF3mc-WDt6NJP8HTzpbhPZZ9RJBPuI/view?usp=sharing)

