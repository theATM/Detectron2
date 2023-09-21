
import os, json, cv2, random


from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from repo.detectron2.engine import DefaultPredictor,  DefaultTrainer
from repo.detectron2.config import get_cfg


def register_datasets(name):
    # register_coco_instances("RSD-GOD", {}, "json_annotation.json", "RSD-GOD")
    register_coco_instances("RSD-COCO-train", {}, "RSD-COCO/annotations/instances_train.json",
                            "RSD-COCO/images/train")
    register_coco_instances("RSD-COCO-val", {}, "RSD-COCO/annotations/instances_val.json",
                            "RSD-COCO/images/val")
    register_coco_instances("RSD-COCO-test", {}, "RSD-COCO/annotations/instances_test.json",
                            "RSD-COCO/images/test")

    # Alternative (with different dir structure):
    #register_coco_instances("RSD-COCO-train", {}, "RSD-COCO/train/_annotations.coco.json",
    #                        "RSD-COCO/train/images")
    #register_coco_instances("RSD-COCO-val", {}, "RSD-COCO/valid/_annotations.coco.json",
    #                        "RSD-COCO/valid/images")
    #register_coco_instances("RSD-COCO-test", {}, "RSD-COCO/test/_annotations.coco.json",
    #                        "RSD-COCO/test/images")

    from detectron2.structures import BoxMode


def train(name, batch_size = 4,workers = 2,freeze=10,lr =  0.00025, epochs = 5, id='0'):

    if not os.path.exists(f"./output{id}"):
        os.makedirs(f"./output{id}")
        print("Created Output Dir")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.OUTPUT_DIR = f"./output{id}"
    cfg.DATASETS.TRAIN = ("RSD-COCO-train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = int((epochs * 7213) / batch_size) # In detectron2, epoch is MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES -  (RSD-COCO : 7213 training images)
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.BACKBONE.FREEZE_AT = freeze
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5   # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()




def eval(name, checkpoint=None, mode='val',id='0'):

    if not os.path.exists(f"./output_{mode}{id}"):
        os.makedirs(f"./output_{mode}{id}")
        print(f"Created Output {mode} Dir")

    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.OUTPUT_DIR = f"./output_{mode}{id}"
    cfg.DATASETS.TEST = (f"RSD-COCO-{mode}",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 8 # This is the real "batch size" commonly known to deep learning people
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    #cfg.MODEL.WEIGHTS = "runs/train4_6000/output/model_final.pth"
    if checkpoint is None:
        cfg.MODEL.WEIGHTS = os.path.join(f"./output{id}", "model_final.pth")  # path to the model we just trained
    else:
        cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator, DatasetEvaluators
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator(f"RSD-COCO-{mode}", output_dir=f"./output_{mode}{id}")
    val_loader = build_detection_test_loader(cfg, f"RSD-COCO-{mode}")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))





def predict(name, checkpoint=None, dataset = None):

    if not os.path.exists("./output_pred"):
        os.makedirs("./output_pred")
        print("Created Output Pred Dir")

    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    if checkpoint is None:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    else:
        cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import build_detection_test_loader
    from detectron2.data import MetadataCatalog, DatasetCatalog
    if dataset is None:
        cfg.DATASETS.TEST = ("RSD-COCO-test",)
        dataset_metadata = MetadataCatalog.get("RSD-COCO-test")
        dataset_dicts = build_detection_test_loader(cfg, "RSD-COCO-test")
    else:
        cfg.DATASETS.TEST = (f"{dataset}",)
        dataset_metadata = MetadataCatalog.get(f"{dataset}")
        dataset_dicts = build_detection_test_loader(cfg, f"{dataset}")




    for d in dataset_dicts:
        im = cv2.imread(d[0]["file_name"])
        cv2.imshow('image', im)
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # "output_pred/"+im,
        cv2.imwrite('./output_pred/'+d[0]["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])




if __name__ == '__main__':
    register_datasets('rsd')

    # TRAINING:

    # Train 10:
    #train('rsd',batch_size = 4,workers = 4,freeze=10,lr =  0.00025, epochs = 5, id='10')
    #eval('rsd',None,'val',id='10')
    #eval('rsd', None, 'test',id='10')

    # Train 11:
    #train('rsd', batch_size = 4, workers = 4, freeze=0, lr =  0.00025, epochs = 5, id='11')
    #eval('rsd', None, 'val', id='11')
    #eval('rsd', None, 'test', id='11')

    # Train 12:
    #train('rsd', batch_size = 8, workers = 8, freeze=10, lr =  0.00025, epochs = 5, id='12')
    #eval('rsd', None, 'val', id='12')
    #eval('rsd', None, 'test', id='12')

    # Train 13:
    #train('rsd', batch_size = 8, workers = 8, freeze=0, lr =  0.00025, epochs = 5, id='13')
    #eval('rsd', None, 'val', id='13')
    #eval('rsd', None, 'test', id='13')

    # Train 14:
    # train('rsd',batch_size = 16,workers = 16,freeze=0,lr =  0.00025, epochs = 5, id='14')
    # eval('rsd',None,'val',id='14')
    # eval('rsd', None, 'test',id='14')

    # Train 15:
    # train('rsd', batch_size = 16, workers = 16, freeze=10, lr =  0.00025, epochs = 5, id='15')
    # eval('rsd', None, 'val', id='15')
    # eval('rsd', None, 'test', id='15')

    # Train 16:
    # train('rsd', batch_size = 8, workers = 8, freeze=5, lr =  0.00025, epochs = 5, id='16')
    # eval('rsd', None, 'val', id='16')
    # eval('rsd', None, 'test', id='16')

    # Train 17:
    # train('rsd', batch_size = 4, workers = 4, freeze=5, lr =  0.00025, epochs = 5, id='17')
    # eval('rsd', None, 'val', id='17')
    # eval('rsd', None, 'test', id='17')

    # Train 18:
    # train('rsd', batch_size = 16, workers = 16, freeze=5, lr =  0.00025, epochs = 5, id='18')
    # eval('rsd', None, 'val', id='18')
    # eval('rsd', None, 'test', id='18')

    # Train 19:
    # train('rsd', batch_size = 16, workers = 16, freeze=5, lr =  0.00025, epochs = 5, id='19')
    # eval('rsd', None, 'val', id='19')
    # eval('rsd', None, 'test', id='19')

    # Train 20:
    # checkpoint = "output20/model_final.pth"
    # train('rsd', batch_size = 4, workers = 8, freeze=0, lr =  0.00025, epochs = 25, id='20')
    # eval('rsd', None, 'val', id='20')
    # eval('rsd', checkpoint, 'test', id='20')


    # EVALUATION:

    #checkpoint = "output20/model_final.pth"
    # eval('rsd', checkpoint, 'val', id='20')
    #eval('rsd', checkpoint, 'test', id='20')


    # PREDICTION on dataset:

    #checkpoint = './output20/model_final.pth'
    #predict('rsd', checkpoint)


