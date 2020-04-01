import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

if __name__ == "__main__":
    cub_data = [[79.41, "Class Proxy"], 
                [78.33, "Sum"], 
                [78.30, "Mean"]]
    cub_std = [0.64, 0.62, 0.60]

    mini_imagenet_data = [[71.02, "Class Proxy"], 
                          [66.36, "Sum"], 
                          [67.47, "Mean"]]
    mini_imagenet_std = [0.62, 0.74, 0.61]

    cub_data = pandas.DataFrame(cub_data, columns = ["Accuracy(%)", "Class Representative"] )
    mini_imagenet_data = pandas.DataFrame(mini_imagenet_data, columns = ["Accuracy(%)", "Class Representative"] )
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0,0], [0,0])
    axes = plt.gca()
    axes.set_ylim([60, 80])
    axes.set_title("CUB")
    sns.barplot(data = cub_data, x="Class Representative", y="Accuracy(%)", yerr=cub_std);
    plt.subplot(1, 2, 2)
    plt.plot([0,0], [0,1])
    axes = plt.gca()
    axes.set_ylim([60, 80])
    axes.set_title("mini-ImageNet")
    sns.barplot(data = mini_imagenet_data, x="Class Representative", y="Accuracy(%)", yerr=mini_imagenet_std);
    plt.savefig("proxy.eps", format='eps')
    plt.show()
