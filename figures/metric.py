import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

if __name__ == "__main__":
    cub_data = [[79.41, "3DConvDistance"], 
            [75.37, "FCDistance"], 
            [73.33, "EuclideanDistance"]]

    mini_imagenet_data = [[71.02, "3DConvDistance"], [66.36, "FCDistance"], [67.47, "EuclideanDistance"]]
    cub_data = pandas.DataFrame(cub_data, columns = ["Score", "Metric"] )
    mini_imagenet_data = pandas.DataFrame(mini_imagenet_data, columns = ["Score", "Metric"] )
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0,0], [0,0])
    axes = plt.gca()
    axes.set_ylim([60, 80])
    axes.set_title("CUB")
    sns.barplot(data = cub_data, x="Metric", y="Score");
    plt.subplot(1, 2, 2)
    plt.plot([0,0], [0,1])
    axes = plt.gca()
    axes.set_ylim([60, 80])
    axes.set_title("mini-ImageNet")
    sns.barplot(data = mini_imagenet_data, x="Metric", y="Score");
    plt.savefig("metric.eps", format='eps')
    plt.show()
