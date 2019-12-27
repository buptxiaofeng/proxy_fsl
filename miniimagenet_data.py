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
    def load_data(self, data_type, num_way, num_shot, num_query):
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
                image = numpy.transpose(image, (2, 0, 1))
                if line[1] not in image_dict:
                    image_dict[line[1]] = [image]
                else:
                    image_dict[line[1]] = image_dict[line[1]] + [image]
            with open(os.path.join("data", "mini_imagenet_" + data_type + "_" + str(num_way) + "_" + str(num_shot) + ".pickle"), "wb") as handle:
                pickle.dump(image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        class_index_list = numpy.random.randint(0, len(image_dict), num_way)

        #data shape is (num_way, num_shot, 3, 84, 84)
        support_image = numpy.zeros((num_way, num_shot, 3, 84, 84), dtype = numpy.float32)
        support_label = numpy.zeros((num_way, num_shot, num_way), dtype=numpy.float32)
        query_image = numpy.zeros((num_way, num_query, 3, 84, 84), dtype = numpy.float32)
        query_label = numpy.zeros((num_way, num_query, num_way), dtype=numpy.float32)

        mean = numpy.array([0.485, 0.456, 0.406])
        mean = numpy.reshape(mean, (3, 1, 1))
        mean = numpy.repeat(mean, 84, axis = 1)
        mean = numpy.repeat(mean, 84, axis = 2)

        std = numpy.array([0.229, 0.224, 0.225])
        std = numpy.reshape(std, (3, 1, 1))
        std = numpy.repeat(std, 84, axis = 1)
        std = numpy.repeat(std, 84, axis = 2)

        for i, class_index in enumerate(class_index_list):
            key = list(image_dict.keys())[class_index]
            image_list = image_dict[key]
            image_index_list = numpy.random.randint(0, len(image_list), num_shot)
            seq_list = list(range(len(image_list)))
            for support_index, tmp_image in enumerate(image_index_list):
                tmp_image = image_list[tmp_image]
                tmp_image = tmp_image - mean
                tmp_image = tmp_image / std
                support_image[i, support_index, ...] = tmp_image
                support_label[i, support_index,i] = 1

            seq_list = list(set(seq_list) - set(image_index_list))
            image_index_list = numpy.random.randint(0, len(seq_list), num_query)

            for query_index, tmp_image in enumerate(image_index_list):
                tmp_image = image_list[seq_list[tmp_image]]
                tmp_image = tmp_image - mean
                tmp_image = tmp_image / std
                query_image[i, query_index, ...] = tmp_image
                query_label[i, query_index,i] = 1

        return support_image, support_label, query_image, query_label

    def _load_from_pickle(self, data_type, num_way, num_shot):
        with open(os.path.join("data", "mini_imagenet_" + data_type + "_" + str(num_way) + "_" + str(num_shot) + ".pickle"), "rb") as handle:
            image_dict = pickle.load(handle)

        return image_dict

if __name__ == "__main__":
    mini = MiniImagenet()
    mini.load_data("train", num_way = 5, num_shot = 5, num_query = 15)
