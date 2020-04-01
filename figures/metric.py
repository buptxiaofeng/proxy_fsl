import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

if __name__ == "__main__":
    cub_data = [[79.41, "3DConvDistance"], 
                [75.22, "FCDistance"], 
                [71.98, "EuclideanDistance"]]
    cub_std = [0.64, 0.67 , 0.71]
    cub_std_str = ["0.64", "0.67" , "0.71"]

    mini_imagenet_data = [[71.02, "3DConvDistance"], 
                          [66.36, "FCDistance"], 
                          [67.46, "EuclideanDistance"]]
    mini_imagenet_std = [0.62, 0.68, 0.70]
    mini_imagenet_std_str = ["0.62", "0.68", "0.70"]

    cub_data = pandas.DataFrame(cub_data, columns = ["Accuracy(%)", "Metric"] )
    mini_imagenet_data = pandas.DataFrame(mini_imagenet_data, columns = ["Accuracy(%)", "Metric"] )
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0,0], [0,0])
    axes = plt.gca()
    axes.set_ylim([60, 85])
    axes.set_title("CUB")
    g = sns.barplot(data = cub_data, x="Metric", y="Accuracy(%)", yerr=cub_std);
    for p, std in zip(g.patches, cub_std_str):
        g.annotate("%.2f" % p.get_height() + u"\u00B1" + std, (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', color='black', xytext=(0, 15),
        textcoords='offset points')
    plt.subplot(1, 2, 2)
    plt.plot([0,0], [0,1])
    axes = plt.gca()
    axes.set_ylim([60, 85])
    axes.set_title("mini-ImageNet")
    g = sns.barplot(data = mini_imagenet_data, x="Metric", y="Accuracy(%)", yerr=mini_imagenet_std);
    for p, std in zip(g.patches, mini_imagenet_std_str):
        g.annotate("%.2f" % p.get_height() + u"\u00B1" + std, (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', color='black', xytext=(0, 15),
        textcoords='offset points')
    plt.savefig("metric.eps", format='eps', bbox_inches='tight')
    plt.show()
