import torch

import torchvision.models as models
import torch.nn as nn


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_model(arch, model_weight, use_gpu):
    if arch.lower().startswith('resnet18'):
        feature_length = 512
    elif arch.lower().startswith('alexnet'):
        feature_length = 4096
    elif arch.lower().startswith('densenet161'):
        feature_length = 2208
    else:
        feature_length = 2048

    model = models.__dict__[arch](num_classes=365)


    if arch.lower().startswith('alexnet'):
        model.features = torch.nn.DataParallel(model.features)

    checkpoint = torch.load(model_weight, map_location=lambda storage, loc: storage)

    if arch.lower().startswith('alexnet'):
        model.load_state_dict(checkpoint['state_dict'])
    elif arch.lower().startswith('densenet161'):
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)
        pass
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

    if arch.lower().startswith('alexnet'):
        model.classifier[6] = nn.Linear(feature_length, 4)
    elif arch.lower().startswith('densenet161'):
        model.classifier = nn.Linear(feature_length, 4)
    else:
        model.fc = nn.Linear(feature_length, 4)

    if use_gpu:
        model = model.cuda()

    print('Loading checkpoint from %s' % model_weight)
    return model
