import sys
import glob
import os
from subprocess import call
from shutil import copyfile

def copy_cub_images(base_path):
    dir_list = glob.glob("CUB_200_2011/images/*")
    for dir_item in dir_list:
        src_path = os.path.join(dir_item, "*.jpg")
        image_list = glob.glob(src_path)
        for image in image_list:
            image_name = image.split("/")[-1]
            copyfile(image, os.path.join("images", image_name))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Incorrect command! e.g., python3 process.py DATASET [cub, miniimagenet]")
    dataset = sys.argv[1]

    if dataset not in ["cub", "miniimagenet"]:
        raise Exception("No such dataset!")

    print("--- process " + dataset + " dataset ---")
    base_path = os.path.join("data", dataset, "images")
    if not os.path.exists(os.path.join("data", dataset, "images")):
        os.makedirs(base_path)
    os.chdir(os.path.join("data", dataset))

    #download files
    if dataset == "cub":
        #call('wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz', shell=True)
        call('tar -zxf CUB_200_2011.tgz', shell=True)
        copy_cub_images(base_path)

    elif dataset == "miniimagenet":
        call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/mini_imagenet_full_size.tar.bz2', shell=True)
        call('tar -xjf mini_imagenet_full_size.tar.bz2', shell=True)
