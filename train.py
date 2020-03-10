import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import numpy
from feat.dataloader.mini_imagenet import MiniImageNet
from feat.dataloader.cub import CUB
from feat.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from feat.models.relation import Relation
from torch.optim.lr_scheduler import StepLR
from evaluation import evaluation
from tqdm import tqdm
import math
from torch.nn import init
import wandb

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

def train():
    json_file = open("parameters.json")
    parameters = json.load(json_file)
    json_file.close()

    train_set = None
    val_set = None
    test_set = None
    if parameters["dataset"] == "CUB":
        train_set = CUB(setname = "train", model_type = parameters["model_type"])
        val_set = CUB(setname = 'val', model_type = parameters["model_type"])
        test_set = CUB(setname = 'test', model_type = parameters["model_type"])
    else:
        train_set = MiniImageNet(setname = "train", model_type = parameters["model_type"])
        val_set = MiniImageNet(setname = 'val', model_type = parameters["model_type"])
        test_set = MiniImageNet(setname = 'test', model_type = parameters["model_type"])

    train_sampler = CategoriesSampler(train_set.label, n_batch = parameters["num_train"], n_cls = parameters["num_way"], n_per = parameters["num_shot"] + parameters["num_query"])
    train_loader = DataLoader(dataset=train_set, batch_sampler = train_sampler, num_workers= 8, pin_memory = True)
    val_sampler = CategoriesSampler(val_set.label, n_batch = parameters["num_val"], n_cls = parameters["num_way"], n_per = parameters["num_shot"] + parameters["num_query"])
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    test_sampler = CategoriesSampler(test_set.label, n_batch = parameters["num_test"], n_cls = parameters["num_way"], n_per = parameters["num_shot"] + parameters["num_query"])
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

    relation = Relation(model_type = parameters["model_type"], num_shot = parameters["num_shot"], num_way = parameters["num_way"], num_query = parameters["num_query"])

    optimizer = torch.optim.SGD(relation.parameters(), lr = parameters["sgd_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience = 50, factor = 0.5, min_lr = 0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience = 20, factor = 0.2, min_lr = 0.0001)
    ce = nn.CrossEntropyLoss().cuda()
    relation = torch.nn.DataParallel(relation, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True

    label = torch.arange(parameters["num_way"]).repeat(parameters["num_query"])
    #one hot encode
    one_hot_label = torch.zeros(len(label), label.max()+1).scatter_(1, label.unsqueeze(1), 1.).float().cuda()
    label = label.cuda()
    max_acc = 0
    max_test_acc = 0
    min_loss = 100
    train_accuracy = 0
    #relation.load_state_dict(torch.load("conv6_best.pth"))
    for epoch in range(parameters["num_epochs"]):
        total_rewards = 0
        total_loss = 0
        for i, batch in enumerate(train_loader,1):
            relation.train()
            data, _ = [_.cuda() for _ in batch]
            p = parameters["num_shot"] * parameters["num_way"]
            support, query = data[:p], data[p:]
            relation_score, support_feature, center  = relation(support, query)

            loss = ce(-1 * relation_score, label) 
            total_loss += loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            
            relation.zero_grad()
            loss.backward()
            optimizer.step()

            episode = epoch * parameters["num_train"] + i + 1
            if episode % 100 == 0:
                print("episode:", epoch * parameters["num_train"] + i+1,"ce loss", total_loss / 100, "contrast loss:", loss2.item())
                train_accuracy = numpy.sum(total_rewards)/1.0/parameters["num_query"] / parameters["num_way"] / parameters["num_train"]
                print('Train Accuracy of the model on the train :{:.2f} %'.format(100 * train_accuracy))
            if (episode % 100 == 0 and episode > 10000) or episode % 1000 == 0:
                acc, _ = evaluation(parameters, relation, val_loader, mode="val")
                if acc > max_acc:
                    test_acc, _, = evaluation(parameters, relation, test_loader, mode="test")
                    torch.save(relation.state_dict(), "conv6_best.pth")
                max_test_acc = max(test_acc, max_test_acc)
                max_acc = max(max_acc, acc)
                print("episode:", epoch * parameters["num_train"] + i+1,"max val acc:", max_acc, " max test acc:", max_test_acc)

        scheduler.step(max_acc)
        print("sgd learning rate:", get_learning_rate(optimizer))

if __name__ == "__main__":
    train()
