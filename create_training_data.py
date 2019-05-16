import os
import pandas as pd
from shutil import copyfile
import argparse
import random

def create_data(input_folder, output_folder):
    #read annotation file
    input_image_folder = os.path.join(input_folder,"images")
    annotations = pd.read_csv(os.path.join(input_folder,"annotations.csv"))


    for k in range(len(annotations['frame'])):
        if k/len(annotations['frame']) * 100 % 2 == 0:
            print("%s Done" % str(k/len(annotations['frame']) * 100 ))
        frame_name = os.path.join(input_image_folder,str(annotations['frame'][k])+".jpg")

        if random.uniform(0, 1) > 0.8:
            destination_folder = os.path.join(output_folder, "eval")
        else:
            destination_folder = os.path.join(output_folder, "train")

        destination =  os.path.join(os.path.join(destination_folder,str(annotations['tamper'][k])), str(annotations['frame'][k])+".jpg")

        #check if file exist
        fileexists = True
        i = 0
        while fileexists:
            if os.path.isfile(destination):
                destination = os.path.join(os.path.join(destination_folder,str(annotations['tamper'][k])), str(annotations['frame'][k])+"_"+str(i)+".jpg")
                i += 1
            else:
                fileexists = False

        copyfile(frame_name,destination)



if __name__== '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--input', default="/media/pmantini/Seagate Backup Plus Drive/Datasets/Camera Tampering Detection/Comprehensive Dataset For Camera Tampering Detection/train/1st floor cam 5/Feb 8th", type=str, help='Input Folder')
    parser.add_argument('--output', default="/media/pmantini/Seagate Backup Plus Drive/dataset_tampering_eval2", type=str, help='Output Folder')
    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    create_data(input_folder, output_folder)