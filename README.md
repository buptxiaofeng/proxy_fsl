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
|Conv-4           |65.25±0.90|80.12±0.62|  52.01±0.82 |  68.37±0.69 |
|Conv-6           |67.29±0.92|82.36±0.57|  52.18±0.82 |  66.17±0.68 |
|ResNet-10        |74.00±0.86|86.03±0.51|  54.18±0.84 |  00.00±0.00 |
|ResNet-18        |75.45±0.92|86.27±0.55|  53.85±0.87 |  00.00±0.00 |
|ResNet-34        |77.70±0.86|87.05±0.52|  52.63±0.90 |  00.00±0.00 |

Table:mini-Imagenet->CUB
|                 |    CUB   |    CUB   |mini-ImageNet|mini-ImageNet|
|-----------------|:--------:|:--------:|:-----------:|:-----------:|
|Embedding Network|  1-shot  |  5-shot  |    1-shot   |    5-shot   |
|Conv-4           |65.25±0.90|80.12±0.62|  52.01±0.82 |  68.37±0.69 |
|Conv-6           |67.29±0.92|82.36±0.57|  52.18±0.82 |  66.17±0.68 |
|ResNet-10        |74.00±0.86|86.03±0.51|  54.18±0.84 |  00.00±0.00 |
|ResNet-18        |75.45±0.92|86.27±0.55|  53.85±0.87 |  00.00±0.00 |
|ResNet-34        |77.70±0.86|87.05±0.52|  52.63±0.90 |  00.00±0.00 |
