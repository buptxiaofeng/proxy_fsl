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
    data = [[80.12, 79.41, 75.29, 76.39, 77.84, 79.34],
            [82.36, 77.24, 77.92, 82.03, 80.16, 82.02],
            [86.03, 76.15, 83.47, 85.01, 83.70, 85.17],
            [86.28, 72.50, 84.45, 86.64, 84.05, 83.58],
            [87.05, 71.83, 86.51, 87.86, 83.18, 84.50]]
    data = pandas.DataFrame(data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["ProxyNet_with_augmentation", "ProxyNet", "MatchingNet", "ProtoNet", "RelationNet", "Baseline++"])
    #data = [[80.12, 79.41],
    #        [82.36, 77.24],
    #        [86.03, 76.15],
    #        [86.28, 72.50],
    #        [87.05, 71.83]]
    #data = pandas.DataFrame(data, index=["Conv-4", "Conv-6", "ResNet-10", "ResNet-18", "ResNet-34"], columns = ["ProxyNet_with_augmentation", "ProxyNet"])
    axes = plt.gca()
    axes.set_ylim([60, 90])
    sns.lineplot(data=data)
    plt.ylabel("Accuracy(%)")
    plt.savefig("augmentation.eps", format='eps', bbox_inches='tight')
    plt.show()
