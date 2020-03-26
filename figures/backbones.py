import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

def backboneplot(data, title):
    axes = plt.gca()
    axes.set_ylim([60, 100])
    axes.set_title(title)
    sns.lineplot(data=data)

if __name__ == "__main__":
    cub_data = [[76.11, 70.77, 72.86, ],
                [77.81, 78.12, 77.08, ],
                [81.12, 84.76, 83.59, ],
                [82.75, 87.42, 83.64, ],
                [82.30, 85.98, 83.78, ],
               ]
    mini_imagenet_data = [[66.60, 64.24, 63.48 ],
                          [64.55, 67.33, 63.19, ],
                          [70.20, 72.64, 68.82, ],
                          [69.83, 73.68, 68.88, ],
                          [69.61, 74.65, 68.32, ]
                         ]
    cub_data = pandas.DataFrame(cub_data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["RelationNet", "ProtoNet", "MatchNet"] )
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0,0], [0,0])
    backboneplot(cub_data, title="CUB")
    mini_imagenet_data = pandas.DataFrame(mini_imagenet_data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["RelationNet", "ProtoNet", "MatchNet"])
    plt.subplot(1, 2, 2)
    plt.plot([0,0], [0,1])
    backboneplot(mini_imagenet_data, title="mini-ImageNet")
    plt.savefig("embedding_functions.eps", format='eps')
    plt.show()
