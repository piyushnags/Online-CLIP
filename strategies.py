from typing import Callable, Sequence, Union
from avalanche.core import BasePlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.templates import SupervisedTemplate
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F



class CLIPSupervisedTemplate(SupervisedTemplate):
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion, labels: str,
                 train_mb_size: int = 1, train_epochs: int = 1, eval_mb_size: int = 1,
                 device: str = "cpu", plugins: Sequence[BasePlugin] = None,
                 evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = default_evaluator,
                 eval_every=-1, peval_mode="epoch"):
        super(CLIPSupervisedTemplate,
              self).__init__(model, optimizer, criterion, 
                             train_mb_size, train_epochs, eval_mb_size, device,
                             plugins, evaluator, eval_every, peval_mode)
        
        self.labels = labels


    @property
    def mb_x(self):
        return self.mbatch[0]


    @property
    def mb_y(self):
        return self.mbatch[1]


    def _unpack_minibatch(self):
        self.mbatch = [item.to(self.device) for item in self.mbatch]
    

    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y)
        return loss
    

    def forward(self):
        logits = self.model(self.mb_x, self.labels)
        return logits