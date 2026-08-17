from tabulate import tabulate


def show_models():
    from semseg import models

    model_names = models.__all__
    numbers = list(range(1, len(model_names)+1))
    print(tabulate({'No.': numbers, 'Model Names': model_names}, headers='keys'))


def show_backbones():
    from semseg.models import backbones

    backbone_names = backbones.__all__
    variants = []
    for name in backbone_names:
        try:
            variants.append(list(eval(f"backbones.{name.lower()}_settings").keys()))
        except:
            variants.append('-')
    print(tabulate({'Backbone Names': backbone_names, 'Variants': variants}, headers='keys'))


def show_heads():
    from semseg.models import heads

    head_names = heads.__all__
    numbers = list(range(1, len(head_names)+1))
    print(tabulate({'No.': numbers, 'Heads': head_names}, headers='keys'))


def show_datasets():
    from semseg import datasets

    dataset_names = datasets.__all__
    numbers = list(range(1, len(dataset_names)+1))
    print(tabulate({'No.': numbers, 'Datasets': dataset_names}, headers='keys'))
