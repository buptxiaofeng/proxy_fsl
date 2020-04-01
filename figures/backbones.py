import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

def backboneplot(data, title):
    axes = plt.gca()
    axes.set_ylim([40, 90])
    axes.set_title(title)
    sns.lineplot(data=data)

def plot(cub_data, mini_imagenet_data, num_shot = 5):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0,0], [0,0])
    backboneplot(cub_data, title="CUB")
    plt.subplot(1, 2, 2)
    plt.plot([0,0], [0,1])
    backboneplot(mini_imagenet_data, title="mini-ImageNet")
    plt.savefig("embedding_functions_"+str(num_shot)+"shot.eps", format='eps')
    plt.show()

if __name__ == "__main__":
    #cub_data = [[79.41, 77.84, 76.39, 75.29, ],
    #            [77.24, 80.16, 82.03, 77.92, ],
    #            [76.15, 83.70, 85.01, 83.47, ],
    #            [72.50, 84.05, 86.64, 84.45, ],
    #            [71.83, 83.18, 87.86, 86.51, ],
    #           ]
    #mini_imagenet_data = [[71.02, 66.60, 64.24, 63.48 ],
    #                      [69.52, 64.55, 67.33, 63.19, ],
    #                      [69.51, 70.20, 72.64, 68.82, ],
    #                      [68.85, 69.83, 73.68, 68.88, ],
    #                      [71.00, 69.61, 74.65, 68.32, ]
    #                     ]
    #cub_data = pandas.DataFrame(cub_data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["ProxyNet", "RelationNet", "ProtoNet", "MatchingNet"] )
    #mini_imagenet_data = pandas.DataFrame(mini_imagenet_data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["ProxyNet", "RelationNet", "ProtoNet", "MatchingNet"])
    #plot(cub_data, mini_imagenet_data, num_shot = 5)

    #cub_data = [[63.95, 62.34, 50.46, 60.52],
    #            [63.22, 64.38, 66.36, 66.47], 
    #            [62.49, 70.47, 73.22, 71.29],
    #            [63.47, 68.58, 72.99, 73.49],
    #            [62.68, 69.72, 72.94, 73.49]]
    #
    #mini_imagenet_data = [[52.10, 49.31, 44.42, 48.14],
    #                      [52.29, 51.84, 50.37, 50.47],
    #                      [53.27, 52.19, 51.98, 54.49],
    #                      [53.64, 52.48, 54.16, 52.91],
    #                      [52.69, 51.74, 53.90, 53.20]]
    #cub_data = pandas.DataFrame(cub_data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["ProxyNet", "RelationNet", "ProtoNet", "MatchingNet"] )
    #mini_imagenet_data = pandas.DataFrame(mini_imagenet_data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["ProxyNet", "RelationNet", "ProtoNet", "MatchingNet"])

    #plot(cub_data, mini_imagenet_data, num_shot = 1)

    data = [[63.95, 79.41, 52.10, 71.02],
            [63.22, 77.24, 52.29, 69.52],
            [62.49, 76.15, 53.27, 69.10],
            [63.47, 72.50, 53.64, 68.85],
            [62.68, 71.83, 52.69, 68.47]]
    data = pandas.DataFrame(data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["CUB_1_shot", "CUB_5_shot", "mini_imagenet_1_shot", "mini_imagenet_5_shot"] )
    axes = plt.gca()
    axes.set_ylim([50, 85])
    sns.lineplot(data=data)
    plt.savefig("embedding_functions.eps", format='eps')
    plt.show()
