import torch.nn as nn
import torch

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

        self.layer1 = nn.Sequential(
                nn.Conv3d(2, 2, kernel_size = 3, padding = 0),
                nn.BatchNorm3d(2),
                nn.ReLU(),
                nn.Conv3d(2, 1, kernel_size = 3, padding = 0),
                nn.BatchNorm3d(1),
                nn.ReLU()
                #nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
                #nn.BatchNorm2d(64),
                #nn.ReLU(),
                #nn.Conv2d(64, 64, kernel_size=3, padding=1),
                #nn.BatchNorm2d(64),
                #nn.ReLU(),
                )

        #self.center = nn.Sequential(
        #        nn.Linear(5, 32),
        #        nn.ReLU(),
        #        nn.Linear(32, 32),
        #        nn.ReLU(),
        #        nn.Linear(32, 1),
        #        nn.ReLU()
        #        )
        self.center = nn.Sequential(
                nn.Conv3d(5, 5, kernel_size = 5, padding = 2),
                nn.BatchNorm3d(1),
                nn.ReLU(),
                nn.Conv3d(5, 1, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(1),
                nn.ReLU()
                )

        #global average pooling
        self.layer2 = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, support, query):

        support = self.encoder(support)
        query = self.encoder(query)
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
        support = torch.sum(support, dim = 1).squeeze()
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
        #out = self.softmax(-1 * out)

        return out, support_out, center
