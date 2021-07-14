import torch.nn as nn

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class MultiStageDetector(BaseDetector):
    def __init__(self, rpn_cfg, refnet_cfg, train_cfg=None, test_cfg=None, pretrained=None,):
        super(MultiStageDetector, self).__init__()
        self.reader = builder.build_reader(rpn_cfg.reader)
        self.backbone = builder.build_backbone(rpn_cfg.backbone)
        if rpn_cfg.neck is not None:
            self.neck = builder.build_neck(rpn_cfg.neck)
        self.bbox_head = builder.build_head(rpn_cfg.bbox_head)
        self.refnet = builder.build_detector(refnet_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.init_weights(pretrained=pretrained)

    # todo: this function seems not used for initialization
    def init_weights(self, pretrained=None):
        super(MultiStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, example):
        x = self.extract_feat(example)
        outs = self.bbox_head(x)
        return outs

    """
    def simple_test(self, example, example_meta, rescale=False):
        x = self.extract_feat(example)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (example_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
    """

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass
