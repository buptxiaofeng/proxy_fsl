import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias = False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias = False),
                nn.Sigmoid()
                )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)

        return out
class SumProxy(nn.Module):
    def __init__(self, dim = 1):
        super(SumProxy, self).__init__()
        self.dim = 1
    
    def forward(self, x):

        return torch.sum(x, dim = self.dim, keepdim = False)

class CosineProxy(nn.Module):
    def __init__(self, num_shot = 5, input_dim = 32):
        super(CosineProxy, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d((input_dim, 5, 5))
        self.consine = nn.CosineSimilarity()

    def forward(self, x):
        out = None
        out_x = torch.sum(x, dim = 1)
        out_x = out_x.squeeze(1)
        out_x = self.pooling(out_x)
        out_x = out_x.view(out_x.shape[0], -1)
        for i in range(x.shape[0]):
            new_x = x[i, ...]
            tmp_x = x[i, ...]
            tmp_out = out_x[i, ...]
            tmp_out = tmp_out.repeat(tmp_x.shape[0], 1)
            tmp_x = self.pooling(tmp_x)
            tmp_x = tmp_x.view(tmp_x.shape[0], -1)
            tmp_x = self.consine(tmp_x, tmp_out)
            shape = new_x.shape
            new_x = torch.mm(tmp_x.unsqueeze(0), new_x.view(new_x.shape[0], -1))
            new_x = new_x.reshape((1, 1, shape[-3], shape[-2], shape[-1]))
            if out is None:
                out = new_x
            else:
                out = torch.cat((out,new_x), dim = 0)

        return out.squeeze(1)

class Proxy(nn.Module):
    def __init__(self, num_shot = 5, input_dim = 32):
        super(Proxy, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d((input_dim, 5, 5))
        self.layer = nn.Sequential(
                nn.Linear(input_dim * 2 * 5 * 5, 32, bias = False),
                nn.ReLU(),
                nn.Linear(32, 1, bias = False),
                nn.Sigmoid()
                )

    def forward(self, x):
        out = None
        out_x = torch.sum(x, dim = 1)
        out_x = out_x.squeeze(1)
        out_x = self.pooling(out_x)
        out_x = out_x.view(out_x.shape[0], -1)
        for i in range(x.shape[0]):
            new_x = x[i, ...]
            tmp_x = x[i, ...]
            tmp_out = out_x[i, ...]
            tmp_out = tmp_out.repeat(tmp_x.shape[0], 1)
            tmp_x = self.pooling(tmp_x)
            tmp_x = tmp_x.view(tmp_x.shape[0], -1)
            tmp_x = torch.cat((tmp_x, tmp_out), dim = 1)
            tmp_x = self.layer(tmp_x)
            tmp_x = tmp_x.squeeze(1)
            shape = new_x.shape
            new_x = torch.mm(tmp_x.unsqueeze(0), new_x.view(new_x.shape[0], -1))
            new_x = new_x.reshape((1, 1, shape[-3], shape[-2], shape[-1]))
            if out is None:
                out = new_x
            else:
                out = torch.cat((out,new_x), dim = 0)

        return out.squeeze(1)

class Relation(nn.Module):

    def __init__(self, model_type, num_shot, num_way, num_query, proxy_type, classifier):
        super(Relation, self).__init__()
        self.num_shot = num_shot
        self.num_way = num_way
        self.num_query = num_query
        self.model_type = model_type
        if model_type == 'ConvNet4':
            from feat.networks.convnet import ConvNet4
            self.encoder = ConvNet4()
            self.input_channels = 64
        elif model_type == 'ConvNet6':
            from feat.networks.convnet import ConvNet6
            self.encoder = ConvNet6()
            self.input_channels = 64
        elif model_type == 'ResNet10':
            from feat.networks.resnet import ResNet10
            self.encoder = ResNet10()
            self.input_channels = 64
        elif model_type == "ResNet18":
            from feat.network.resnet import ResNet18
            self.encoder = ResNet18()
            self.input_channels = 64
        elif model_type == "ResNet34":
            from feat.network.resnet import ResNet34
            self.encoder = ResNet34()
            self.input_channels = 64
        else:
            raise ValueError('')

        if proxy_type == "Cosine":
            self.proxy = CosineProxy(num_shot = self.num_shot)
        elif proxy_type == "Sum":
            self.proxy = SumProxy(dim = 1)
        elif proxy_type == "Proxy":
            self.proxy = Proxy(num_shot = self.num_shot)
        else:
            raise ValueError("")

        self.layer1 = nn.Sequential(
                nn.Conv3d(2, 4, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(4),
                nn.ReLU(),
                nn.Conv3d(4, 1, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(1),
                nn.ReLU(),
                )

        self.se = SELayer(self.input_channels)

        #global average pooling
        self.layer2 = nn.AdaptiveAvgPool3d(1)

    def forward(self, support, query):

        support = self.encoder(support)
        support = self.se(support)

        query = self.encoder(query)
        query = self.se(query)

        shape = support.shape
        support = support.reshape(self.num_shot, self.num_way, support.shape[1] , support.shape[2] , support.shape[3])
        support = torch.transpose(support, 0, 1)

        #for one shot
        if support.shape[1] == 1:
            support = support.squeeze(1)
        else:
            support = self.proxy(support)

        support = support.unsqueeze(0).repeat(self.num_query * self.num_way,1,1,1,1)
        query = query.unsqueeze(0).repeat(self.num_way, 1, 1, 1, 1)
        query = torch.transpose(query, 0, 1)

        support = support.reshape(-1, support.shape[2], support.shape[3], support.shape[4])
        query = query.reshape(-1, query.shape[2], query.shape[3], query.shape[4])

        support = support.unsqueeze(1)
        query = query.unsqueeze(1)

        feature = torch.cat((support, query), 1)
        out = self.layer1(feature)
        out = self.layer2(out)
        out = out.view(-1, self.num_way)

        return out
