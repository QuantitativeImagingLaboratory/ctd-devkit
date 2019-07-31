import argparse
import torch
import pickle
import cv2
import csv
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import numpy as np

def get_model(model_weights, use_gpu):
    if use_gpu:
        model = torch.load(model_weights, pickle_module=pickle)
    else:
        model = torch.load(model_weights, map_location=lambda storage, loc: storage, pickle_module=pickle)

    for param in model.parameters():
        param.requires_grad = False
    print("Loading model from {}".format(model_weights))
    return model

class SceneDatasetVideo(Dataset):
    def __init__(self, video_file, transform=None):
        self.video_capture = cv2.VideoCapture(video_file)
        ret, cv_read_image = self.video_capture.read()

        cv2_im = cv2.cvtColor(cv_read_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_im)
        self.img_size = image.size

        self.transform = transforms.Compose([
                transforms.RandomSizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])


    def __len__(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_next_frame_number(self):
        return self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)

    def get_frame_size(self):
        return (self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, idx):
        ret, cv_read_image = self.video_capture.read()

        cv2_im = cv2.cvtColor(cv_read_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_im)

        if self.transform:
            image = self.transform(image)

        return image, self.get_next_frame_number()-1


def get_dataset(video_file):
    dataset_test = SceneDatasetVideo(video_file=video_file)
    return DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--weights', type=str, help='Weights file')
    parser.add_argument('--video_file', type=str, help='Video file to predict')
    parser.add_argument('--output_file', default='output.csv', type=str, help='Output file')

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    print('Building Model')
    model = get_model(args.weights, use_gpu)


    video = SceneDatasetVideo(args.video_file)

    dataloader_test = get_dataset(args.video_file)

    results = []
    for data in tqdm(dataloader_test):

        inputs, img_ids = data

        if use_gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs)

        outputs = model(inputs)

        for img_id, prob in zip(img_ids, outputs.data):
            results += [[int(img_id.item()), np.argmax(prob.cpu()).item()]]

        if len(results) >= 1000:
            writecsv(args.output_file, results)
            results = []


def writecsv(outputfilename, data):
    with open(outputfilename, 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(data)

if __name__ == "__main__":
    global batch_size
    batch_size = 16
    main()