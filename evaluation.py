import torch
import scipy.stats
import numpy
from feat.dataloader.mini_imagenet import MiniImageNet
from feat.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from feat.models.proxynet import ProxyNet 
import torch.nn as nn
import json
from tqdm import tqdm

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def evaluation(config, proxynet, data_loader, mode = "test"):
    assert mode == "test" or mode == "val"
    num_total = 0
    if mode == "test":
        num_total = config.num_way * config.num_test * config.num_query

    if mode == "val":
        num_total = config.num_way * config.num_val * config.num_query

    ce = nn.CrossEntropyLoss().cuda()

    proxynet.eval()

    acc_list = []

    label = torch.arange(config.num_way).repeat(config.num_query).cuda()
    with torch.no_grad():
        total_rewards = 0
        total_loss = 0
        for batch, _ in tqdm(data_loader):
            data = batch.cuda()
            k = config.num_way * config.num_shot
            support, query = data[:k], data[k:]
            relation_score = proxynet(support, query)
            loss = ce(-1 * relation_score, label)
            total_loss += loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            acc = numpy.sum(rewards) / 1.0 / config.num_way / config.num_query
            acc_list.append(acc)

        m, h = mean_confidence_interval(acc_list)
        print('Test mean accuracy of the model on the', mode, ' :{:.2f} %'.format(m * 100), "interval:", h * 100, "val ce loss:", total_loss / len(data_loader))
    return m, h
