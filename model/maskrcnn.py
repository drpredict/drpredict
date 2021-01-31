from torchvision.models.detection import MaskRCNN as MaskRCNN_
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
import torch

class MaskRCNN(MaskRCNN_):
    def __init__(self, num_class=10, snap=None, trainable_layers=5):
        backbone = resnet_fpn_backbone('resnet101', True, trainable_layers=trainable_layers)
        anchor_sizes = ((8, 16, 32, 64, 128), )
        aspect_ratios = [(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))]
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        super(MaskRCNN, self).__init__(
            backbone,
            num_class,
            # rpn_anchor_generator=rpn_anchor_generator,
        )
        if snap is not None:
            state_dict = torch.load(open(self.snap, 'rb'))
            for k in list(state_dict.keys()):
                if k not in self.state_dict():
                    continue
                if self.state_dict()[k].shape != state_dict[k].shape:
                    print(f'removing key {k}')
                    del state_dict[k]
            # del state_dict['roi_heads.box_predictor.cls_score.weight']
            # del state_dict['roi_heads.box_predictor.cls_score.bias']
            # del state_dict['roi_heads.box_predictor.bbox_pred.weight']
            # del state_dict['roi_heads.box_predictor.bbox_pred.bias']
            # del state_dict['roi_heads.mask_predictor.mask_fcn_logits.weight']
            # del state_dict['roi_heads.mask_predictor.mask_fcn_logits.bias']
            unused = self.load_state_dict(state_dict, strict=False)
            # print("### unused  parameters ", unused)

