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

class Proxy(nn.Module):
    def __init__(self, num_shot = 5):
        super(Proxy, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d((64, 5, 5))
        self.layer = nn.Sequential(
                nn.Linear(64 * 5 * 5, 32, bias = False),
                nn.ReLU(),
                nn.Linear(32, 1, bias = False),
                nn.Sigmoid()
                )

    def forward(self, x):
        out = None
        for i in range(x.shape[0]):
            new_x = x[i, ...]
            tmp_x = x[i, ...]
            tmp_x = self.pooling(tmp_x)
            tmp_x = tmp_x.view(tmp_x.shape[0], -1)
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

    def __init__(self, model_type, num_shot, num_way, num_query):
        super(Relation, self).__init__()
        self.num_shot = num_shot
        self.num_way = num_way
        self.num_query = num_query
        self.model_type = model_type
        if model_type == 'ConvNet4':
            from feat.networks.convnet import ConvNet4
            self.encoder = ConvNet4()
            self.input_channels = 128
        elif model_type == 'ConvNet6':
            from feat.networks.convnet import ConvNet6
            self.encoder = ConvNet6()
            self.input_channels = 128
        elif model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
            self.input_channels = 1280
        else:
            raise ValueError('')

        self.proxy = Proxy()
        self.dropout = nn.Dropout(0.3)

        self.layer1 = nn.Sequential(
                nn.Conv3d(2, 2, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(2),
                nn.ReLU(),
                nn.Conv3d(2, 1, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(1),
                nn.ReLU(),
                )

        self.se = SELayer(64)

        #global average pooling
        self.layer2 = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, support, query):

        support = self.encoder(support)
        support = self.se(support)

        query = self.encoder(query)
        query = self.se(query)
        query_out = query

        shape = support.shape
        support = support.reshape(self.num_shot, self.num_way, support.shape[1] , support.shape[2] , support.shape[3])
        support = torch.transpose(support, 0, 1)
        support_out = support

        #support  = self.center(support).squeeze()

        #support = support.reshape(self.num_shot, self.num_way, support.shape[1] * support.shape[2] * support.shape[3])
        #new_support = None
        #for i in range(self.num_way):
        #    sub_support = support[:, i, ...]
        #    sub_support = torch.transpose(sub_support, 0, 1)
        #    sub_support = self.center(sub_support).squeeze()
        #    sub_support = sub_support.reshape(1, shape[1], shape[2], shape[3])
        #    if new_support is None:
        #        new_support = sub_support
        #    else:
        #        new_support = torch.cat((new_support, sub_support), dim = 0)
        #support = new_support

        #support = support.squeeze().reshape(self.num_way, shape[1], shape[2], shape[3])
        support = self.proxy(support)
        #support = torch.sum(support, dim = 1).squeeze()
        #support = support[:,0,...]
        #support = torch.mean(support, dim = 1).squeeze()
        center = support

        support = support.unsqueeze(0).repeat(self.num_query * self.num_way,1,1,1,1)
        query = query.unsqueeze(0).repeat(self.num_way, 1, 1, 1, 1)
        query = torch.transpose(query, 0, 1)

        #feature = support * support
        support = support.reshape(-1, support.shape[2], support.shape[3], support.shape[4])
        query = query.reshape(-1, query.shape[2], query.shape[3], query.shape[4])

        support = support.unsqueeze(1)
        query = query.unsqueeze(1)

        feature = torch.cat((support, query), 1)
        out = self.layer1(feature)
        out = self.layer2(out)
        out = out.view(-1, self.num_way)

        return out, support_out, center
