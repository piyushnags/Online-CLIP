from typing import Dict
import warnings
import torch
from torch.utils.data import DataLoader
from avalanche.training.plugins import EWCPlugin
from avalanche.training.utils import zerolike_params_dict, ParamData 



class ClipEWCPlugin(EWCPlugin):
    def __init__(
        self,
        ewc_lambda,
        labels,
        mode='separate',
        decay_factor=None,
        keep_importance_data=False,
    ):
        super(ClipEWCPlugin, self).__init__(
            ewc_lambda, 
            mode, 
            decay_factor, 
            keep_importance_data
        )

        self.labels = labels
    

    def compute_importances(self, model, criterion, optimizer, dataset,
                            device, batch_size, num_workers=0) -> Dict[str, ParamData]:
        model.eval()

        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        importances = zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Modified this part to use OpenAI module WITH labels
            out = model(x, self.labels)

            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances