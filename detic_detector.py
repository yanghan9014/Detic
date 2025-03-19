import sys
import cv2
# import tempfile
from pathlib import Path
# import cog
import time
import torch

# import some common detectron2 utilities
# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
import pdb

class Predictor:
    def __init__(self,):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = 'Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        self.model = build_model(cfg)
        self.model.eval()
        # if len(cfg.DATASETS.TEST):
        #     self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }
        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }
        
    def set_vocabulary(self, vocabulary, custom_vocabulary):
        if not vocabulary == 'custom':
            metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[vocabulary]
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.model, classifier, num_classes)
        else:
            assert custom_vocabulary is not None and len(custom_vocabulary.split(',')) > 0, \
                "Please provide your own vocabularies when vocabulary is set to 'custom'."
            metadata.thing_classes = custom_vocabulary.split(',')
            classifier = self._get_clip_embeddings(metadata.thing_classes)
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.model, classifier, num_classes)
            # Reset visualization threshold
            output_score_threshold = 0.3
            for cascade_stages in range(len(self.model.roi_heads.box_predictor)):
                self.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    
    def predict(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        return outputs

    def _get_clip_embeddings(self, vocabulary, prompt='a '):
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def predict_demo(self, image_path, vocabulary, custom_vocabulary):
        image = cv2.imread(str(image_path))
        if not vocabulary == 'custom':
            metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[vocabulary]
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.model, classifier, num_classes)

        else:
            assert custom_vocabulary is not None and len(custom_vocabulary.split(',')) > 0, \
                "Please provide your own vocabularies when vocabulary is set to 'custom'."
            metadata = MetadataCatalog.get(str(time.time()))
            metadata.thing_classes = custom_vocabulary.split(',')
            classifier = self._get_clip_embeddings(metadata.thing_classes)
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.model, classifier, num_classes)
            # Reset visualization threshold
            output_score_threshold = 0.3
            for cascade_stages in range(len(self.model.roi_heads.box_predictor)):
                self.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

        outputs = self.predict(image)
        v = Visualizer(image[:, :, ::-1], metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path = "out.png"
        cv2.imwrite(str(out_path), out.get_image()[:, :, ::-1])
        return out_path
if __name__ == "__main__":
    predictor = Predictor()
    # custom_vocabulary = 'keyboard, mouse, monitor, desk, chair, mug'
    # predictor.predict_demo(Path('desk.png'), 'custom', custom_vocabulary)
    predictor.predict_demo(Path('../../../VidVRD-helper/vidor-dataset/frames/0006/3182270827/0001.jpg'), 'lvis', None)
    # /home/daniel/VidVRD-helper/vidor-dataset/frames/0006/2569620797/0003.jpg
    # /home/daniel/VidVRD-helper/vidor-dataset/frames/0006/3182270827/0001.jpg