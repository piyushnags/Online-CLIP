from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor

from CLIP import clip



class CLIPClassifier(nn.Module):
    def __init__(self, arch: str, device: Any, num_classes: int, jit: bool = False):
        super(CLIPClassifier, self).__init__()

        self.model, self.preprocess = clip.load(arch, device, jit=jit)

    
    def forward(self, imgs: List[Tensor], labels: List[Tensor]
                ) -> List[Tensor]:
        image_feats = self.model.encode_image(imgs)
        text_feats = self.model.encode_text(labels)

        image_feats_norm = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)

        logits = (image_feats_norm @ text_feats_norm.T) * self.model.logit_scale.exp()
        return logits



if __name__ == '__main__':
    pass