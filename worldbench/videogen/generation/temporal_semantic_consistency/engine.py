import os
import numpy as np
from torchvision import transforms
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
import sys
sys.path.append('/home/alan/AlanLiang/Projects/AlanLiang/WorldBench/Code/OpenSeeD')
from utils.arguments import load_opt_command_dict
from openseed.BaseModel import BaseModel
from openseed import build_model


def build_engine(args):
    opt, _ = load_opt_command_dict(args)

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    thing_classes = ['car','person','traffic light', 'truck', 'motorcycle', 'barrier', 'bicycle', 'bus', 'construction_vehicle', 'traffic_cone', 'trailer', 'terrain']
    stuff_classes = ['building','sky','street','tree','rock','sidewalk', 'road', 'vegetation']
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    return model, transform, metadata

if __name__ == "__main__":
    build_engine(args=None)