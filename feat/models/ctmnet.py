import torch.nn as nn
import torch
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FCClassifier(nn.Module):
    def __init__(self, input_size = 64, hidden_size = 8):
        super(FCClassifier, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(64 * 2, 64, kernel_size = 3, padding = 0),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64,kernel_size=3,padding=0),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.fc1 = nn.Linear(input_size*3*3, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out

class CTMNet(nn.Module):

    def __init__(self, model_type, num_shot, num_way, num_query):
        super(CTMNet, self).__init__()
        self.num_shot = num_shot
        self.num_way = num_way
        self.num_query = num_query
        self.model_type = model_type
        if model_type == 'ConvNet4':
            from feat.networks.convnet import ConvNet4
            self.encoder = ConvNet4(pooling = False)
            self.input_channels = 64
        elif model_type == 'ConvNet6':
            from feat.networks.convnet import ConvNet6
            self.encoder = ConvNet6()
            self.input_channels = 64
        elif model_type == 'ResNet10':
            from feat.networks.resnet import ResNet10
            self.encoder = ResNet10()
            self.input_channels = 512
        elif model_type == "ResNet18":
            from feat.networks.resnet import ResNet18
            self.encoder = ResNet18()
            self.input_channels = 512
        elif model_type == "ResNet34":
            from feat.networks.resnet import ResNet34
            self.encoder = ResNet34()
            self.input_channels = 512
        else:
            raise ValueError('')

        self.concentrator = self._make_layer(Bottleneck, 64, 16, 4, stride=1)
        self.projection = self._make_layer(Bottleneck, 320, 16, 4, stride=1)
        self.reshaper = self._make_layer(Bottleneck, 64, 16, 4, stride=1)

        self.classifier = FCClassifier()

    def forward(self, support, query):
        support = self.encoder(support)
        query = self.encoder(query)

        #Concentrator
        _input_P = self.concentrator(support)
        shape = _input_P.shape
        _input_P = support.reshape(self.num_shot, self.num_way, shape[1] , shape[2] , shape[3])
        _input_P = torch.transpose(_input_P, 0, 1)
        if _input_P.shape[1] == 1:
            _input_P = support.squeeze(1)
        else:
            _input_P = torch.mean(_input_P, dim = 1, keepdim = False)

        #Projector
        _input_P = _input_P.reshape(1, _input_P.shape[0] * _input_P.shape[1], _input_P.shape[2], _input_P.shape[3])
        _input_P = self.projection(_input_P)
        P = F.softmax(_input_P, dim = 1)

        #We use matmul here since the original code of CTM is using matmul
        support = self.reshaper(support)
        support = torch.matmul(support, P)
        query = self.reshaper(query)
        query = torch.matmul(query, P)

        #relation net
        support = support.reshape(self.num_shot, self.num_way, support.shape[1] , support.shape[2] , support.shape[3])
        support = torch.transpose(support, 0, 1)

        new_out = None
        for i  in range(self.num_shot):
            tmp_support = support[:, i, ...]
            tmp_query = query
            tmp_support = tmp_support.unsqueeze(0).repeat(self.num_query * self.num_way, 1, 1, 1, 1)
            tmp_query = tmp_query.unsqueeze(0).repeat(self.num_way, 1, 1, 1, 1)
            tmp_query = torch.transpose(tmp_query, 0, 1)

            tmp_support = tmp_support.reshape(-1, tmp_support.shape[2], tmp_support.shape[3], tmp_support.shape[4])
            tmp_query = tmp_query.reshape(-1, tmp_query.shape[2], tmp_query.shape[3], tmp_query.shape[4])

            feature = torch.cat((tmp_support, tmp_query), 1)
            out = self.classifier(feature)
            out = out.view(-1, self.num_way)

            if new_out is None:
                new_out = out
            else:
                new_out = new_out + out

        return new_out / self.num_shot

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
