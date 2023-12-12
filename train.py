from typing import Any

import torch
import torch.nn as nn

from avalanche.benchmarks.generators import nc_benchmark
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import(
    forgetting_metrics, accuracy_metrics,
    loss_metrics, timing_metrics, confusion_matrix_metrics,
    topk_acc_metrics,
)

from strategies import CLIPSupervisedTemplate
from plugins import ClipEWCPlugin
from dataset import get_cifar100
from utils import parse
from CLIP import clip
from models import CLIPClassifier



def train(args: Any):
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Need to set jit=False and set model to fp32 since weights are
    # very prone to under/overflow. jit has hardcoded dtypes too
    # NOTE: Can consider refactoring code to use torch mixed-precision 
    # at a later time 
    # See https://github.com/openai/CLIP/issues/40
    model = CLIPClassifier('ViT-B/16', device, 100, args.jit)
    if not args.jit:
        model.float()

    train_set, test_set = get_cifar100()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    scenario = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=args.n_exp,
        train_transform=model.preprocess,
        eval_transform=model.preprocess,
        task_labels=False,
        seed=args.seed
    )

    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open('logfile.txt', 'a'))
    loggers = [interactive_logger, text_logger]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=args.num_classes, stream=True),
        topk_acc_metrics(minibatch=True, epoch=True, experience=True),
        strict_checks=False,
        loggers=loggers
    )

    labels = torch.cat([ clip.tokenize( f'A bad photo of a {c}' for c in train_set.classes ) ]).to(device)
    plugins = None
    if args.use_ewc:
        ewc_plugin = ClipEWCPlugin(ewc_lambda=args.ewc_l,labels=labels, mode='online',
                                decay_factor=args.ewc_decay, keep_importance_data=False)
        plugins = [ewc_plugin]

    cl_strategy = CLIPSupervisedTemplate(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=args.mb_size,
        train_epochs=args.epochs,
        eval_mb_size=args.mb_size,
        device=device,
        evaluator=eval_plugin,
        plugins=plugins,
        labels=labels
    )

    results = []
    for experience in scenario.train_stream:
        print('Start of experience: ', experience.current_experience)
        print('Current classes: ', experience.classes_in_this_experience)

        # NOTE: Commented out train function to reproduce results from
        # the paper: https://arxiv.org/pdf/2210.03114.pdf
        cl_strategy.train(experience)
        results.append( cl_strategy.eval(scenario.test_stream) )



if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    args = parse()
    train(args)