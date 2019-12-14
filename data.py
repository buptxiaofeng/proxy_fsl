import glob
import pandas
import os
import cv2
from tqdm import tqdm
import pickle
import numpy
from collections import OrderedDict

class MiniImagenet():

    def __init__(self):
        pass

    #data_type should be train, test, val
    def load_data(self, data_type, num_way, num_shot):
        assert data_type == "train" or data_type == "test" or data_type == "val"
        
        image_dict = OrderedDict()
        if os.path.exists(os.path.join("data", "mini_imagenet_" + data_type + "_" + str(num_way) + "_" + str(num_shot) + ".pickle")):
            image_dict = self._load_from_pickle(data_type, num_way, num_shot)
        else:
            csv_path = os.path.join("data", "mini_imagenet")
            csv_path = os.path.join(csv_path, data_type + ".csv")
            csv = pandas.read_csv(csv_path)
            csv = csv.values
            base_path = os.path.join("data", "mini_imagenet", "images")
            for line in tqdm(csv):
                image_path = os.path.join(base_path, line[0])
                image = cv2.imread(image_path)
                image= cv2.resize(image, (84, 84))
                if line[1] not in image_dict:
                    image_dict[line[1]] = [image]
                else:
                    image_dict[line[1]] = image_dict[line[1]] + [image]
            with open(os.path.join("data", "mini_imagenet_" + data_type + "_" + str(num_way) + "_" + str(num_shot) + ".pickle"), "wb") as handle:
                pickle.dump(image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        class_index_list = numpy.random.randint(0, len(image_dict), num_way)

        #data shape is (num_way, num_shot, 84, 84, 3)
        support_image = numpy.zeros((num_way, num_shot, 84, 84, 3))
        support_label = []
        query_image = numpy.zeros((num_way, len(image_dict[list(image_dict.keys())[0]]) - num_shot, 84, 84, 3))
        query_label = []

        for i, class_index in enumerate(class_index_list):
            key = list(image_dict.keys())[class_index]
            support_label = support_label + [key]
            query_label = query_label + [key]
            image_list = image_dict[key]
            image_index_list = numpy.random.randint(0, len(image_list), num_shot)
            support_index = 0
            query_index = 0
            for image_index, tmp_image in enumerate(image_list):
                if image_index in image_index_list:
                    support_image[i, support_index, ...] = tmp_image
                    support_index = support_index + 1
                else:
                    query_image[i, query_index, ...] = tmp_image
                    query_index = query_index + 1

        return support_image, support_label, query_image, query_label

    def _load_from_pickle(self, data_type, num_way, num_shot):
        with open(os.path.join("data", "mini_imagenet_" + data_type + "_" + str(num_way) + "_" + str(num_shot) + ".pickle"), "rb") as handle:
            image_dict = pickle.load(handle)

        return image_dict


if __name__ == "__main__":
    mini = MiniImagenet()
    mini.load_data("train", num_way = 5, num_shot = 5)
