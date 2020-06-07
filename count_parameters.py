from feat.models.proxynet import ProxyNet
from feat.models.matchnet import MatchNet

proxynet = ProxyNet(model_type = "ConvNet4", num_shot = 5, num_way = 5, num_query = 15, proxy_type = "Proxy", classifier = "3DConv")
num_parameters = sum(p.numel() for p in proxynet.parameters() if p.requires_grad)
print("The number of trainable parameter: ProxyNet:" + str(num_parameters))

protonet = ProxyNet(model_type = "ConvNet4", num_shot = 5, num_way = 5, num_query = 15, proxy_type = "Mean", classifier = "Euclidean")
num_parameters = sum(p.numel() for p in protonet.parameters() if p.requires_grad)
print("The number of trainable parameter: ProtoNet:" + str(num_parameters))

relationnet = ProxyNet(model_type = "ConvNet4", num_shot = 5, num_way = 5, num_query = 15, proxy_type = "Sum", classifier = "FC")
num_parameters = sum(p.numel() for p in relationnet.parameters() if p.requires_grad)
print("The number of trainable parameter: RelationNet:" + str(num_parameters))

matchnet = MatchNet(model_type = "ConvNet4", use_bilstm = True, num_shot = 5, num_way = 5, num_query = 15)
num_parameters = sum(p.numel() for p in matchnet.parameters() if p.requires_grad)
print("The number of trainable parameter: MatchNet:" + str(num_parameters))
