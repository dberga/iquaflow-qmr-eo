import argparse
import json
import os
from shutil import copyfile

"""
class SR:
    def apply():
        ...
"""
current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_images_folder = "generated_sr_images"
base_ds = os.path.join(current_path, "test_datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainds", default=os.path.join(base_ds, "inria-aid_short")) #ds_coco_dataset
    parser.add_argument(
        "--traindsinput", default=os.path.join(base_ds, "inria-aid_short", "train/images_short") #ds_coco_dataset, images
    )  # default subforlder from task ds (ds_wrapper.data_input)
    parser.add_argument("--valds", default=os.path.join(base_ds, "inria-aid_short")) #ds_coco_dataset
    parser.add_argument(
        "--valdsinput", default=os.path.join(base_ds, "inria-aid_short", "test/images_short") #ds_coco_dataset, images
    )
    parser.add_argument("--outputpath", default=os.path.join(current_path, "tmp/"))

    args = parser.parse_args()
    train_ds = args.trainds
    train_ds_input = args.traindsinput
    val_ds = args.valds
    val_ds_input = args.valdsinput
    output_path = args.outputpath

    # MOCK SR: create generated_sr_images from copy

    # read train_ds and val_ds files and apply super-resolution, save in output_path/val/generated_sr_images folder

    train_ds_output = os.path.join(
        output_path, "train", output_images_folder
    )  # train_ds_input
    val_ds_output = os.path.join(
        output_path, "val", output_images_folder
    )  # val_ds_input

    # create subfolders
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, "train")):
        os.mkdir(os.path.join(output_path, "train"))
    if not os.path.exists(os.path.join(output_path, "val")):
        os.mkdir(os.path.join(output_path, "val"))

    if not os.path.exists(train_ds_output):
        os.mkdir(train_ds_output)
    if not os.path.exists(val_ds_output):
        os.mkdir(val_ds_output)

    # to do: SR.apply()
    # generate / (mock sr=copy images)
    image_files = os.listdir(train_ds_input)
    for idx, image_name in enumerate(image_files):
        copyfile(train_ds_input + "/" + image_name, train_ds_output + "/" + image_name)
    image_files = os.listdir(val_ds_input)
    for idx, image_name in enumerate(image_files):
        copyfile(val_ds_input + "/" + image_name, val_ds_output + "/" + image_name)

    """
    #alternative (old): copy input image paths as output image paths
    train_ds_output=train_ds_input
    val_ds_output=val_ds_input
    """
    output_json = {
        "train_ds": train_ds,
        "train_ds_input": train_ds_input,
        "train_ds_output": train_ds_output,
        "val_ds": val_ds,
        "val_ds_input": val_ds_input,
        "val_ds_output": val_ds_output,
    }

    with open(os.path.join(output_path, "output.json"), "w") as f:
        json.dump(output_json, f)
