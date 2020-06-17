# proxynet_fsl

|parameter        |          |             |             |             |             |
|:---------------:|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|
|model_type       |ConvNet4  |ConvNet6     |   ResNet10  |   ResNet18  |   ResNet34  |
|dataset          |CUB       |MiniImageNet |             |             |             |
|if_augmentation  |True      |False        |             |             |             |
|num_epochs       |600       |             |             |             |             |
|batch_size       |1         |             |             |             |             |
|sgd_lr           |0.1       |             |             |             |             |
|num_way          |5         |             |             |             |             |
|num_shot         |5         |1            |             |             |             |
|num_query        |16        |             |             |             |             |
|num_train        |100       |             |             |             |             |
|num_val          |600       |             |             |             |             |
|num_test         |600       |             |             |             |             |
|proxy_type       |Proxy     |Mean         |     Sum     |             |             |
|Classifier       |3DConv    |FC           |  Euclidean  |             |             |

Table:The performance of ProxyNet with data augmentation
|                 |    CUB   |    CUB   |mini-ImageNet|mini-ImageNet|
|-----------------|:--------:|:--------:|:-----------:|:-----------:|
|Embedding Network|  1-shot  |  5-shot  |    1-shot   |    5-shot   |
|Conv-4           |67.52±0.97|82.85±0.60|  52.95±0.76 |  70.35±0.63 |
|Conv-6           |68.16±0.93|83.57±0.58|  52.18±0.82 |  69.91±0.62 |
|ResNet-10        |76.79±0.84|88.02±0.52|  54.18±0.84 |  75.27±0.65 |
|ResNet-18        |76.72±0.90|88.63±0.49|  53.85±0.87 |  74.81±0.64 |
|ResNet-34        |77.70±0.86|87.05±0.52|  52.63±0.90 |  74.10±0.61 |
