import torch
import scipy.stats
import numpy
from feat.dataloader.mini_imagenet import MiniImageNet
from feat.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from feat.models.relation import Relation
import torch.nn as nn
import json
from tqdm import tqdm

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def evaluation(parameters, relation, data_loader, mode = "test"):
    assert mode == "test" or mode == "val"
    num_total = 0
    if mode == "test":
        num_total = parameters["num_way"] * parameters["num_test"] * parameters["num_query"]

    if mode == "val":
        num_total = parameters["num_way"] * parameters["num_val"] * parameters["num_query"]

    ce = nn.CrossEntropyLoss().cuda()

    relation.eval()

    acc_list = []

    label = torch.arange(parameters["num_way"]).repeat(parameters["num_query"]).cuda()
    with torch.no_grad():
        total_rewards = 0
        total_loss = 0
        for batch, _ in tqdm(data_loader):
            data = batch.cuda()
            k = parameters["num_way"] * parameters["num_shot"]
            support, query = data[:k], data[k:]
            relation_score = relation(support, query)
            loss = ce(-1 * relation_score, label)
            total_loss += loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            acc = numpy.sum(rewards) / 1.0 / parameters["num_way"] / parameters["num_query"] 
            acc_list.append(acc)

        m, h = mean_confidence_interval(acc_list)
        print('Test mean accuracy of the model on the', mode, ' :{:.2f} %'.format(m * 100), "interval:", h * 100, "val ce loss:", total_loss / len(data_loader))
    return m, h

if __name__ == "__main__":
    json_file = open("parameters.json")
    parameters = json.load(json_file)
    json_file.close()

    test_set = None
    if parameters["dataset"] == "CUB":
        test_set = CUB(setname = 'test', image_size = image_size)
    elif parameters["dataset"] == "MiniImageNet":
        test_set = MiniImageNet(setname = 'test', image_size = image_size)
    save_name = str(parameters["model_type"]) + "_" + str(parameters["dataset"]) + "_" + str(parameters["num_shot"]) + "_" + str(parameters["num_way"]) + ".pth"
    proxynet = Relation(model_type = parameters["model_type"], num_shot = parameters["num_shot"], num_way = parameters["num_way"], num_query = parameters["num_query"], proxy_type = parameters["proxy_type"], classifier = parameters["classifier"]).cuda()
    proxynet.load_state_dict(torch.load(os.path.join("weights", save_name)))
    evaluation(parameters, proxynet, test_set)
