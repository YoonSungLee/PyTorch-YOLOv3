# python test_all.py --model_def config/yolov3-custom.cfg --data_config config/custom.data

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    # checkpoints/yolov3_ckpt_99.pth
    mAP_list = []
    AP_list = []
    for idx in range(100):
        print('%d/100'%idx)
        weights_path = 'checkpoints/yolov3_ckpt_' + str(idx) + '.pth'
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
        parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
        parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
        # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
        parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
        parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        opt = parser.parse_args()
        # print(opt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_config = parse_data_config(opt.data_config)
        valid_path = data_config["valid"]
        class_names = load_classes(data_config["names"])

        # Initiate model
        model = Darknet(opt.model_def).to(device)
        if weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path))

        print("Compute mAP...")

        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=8,
        )

        mAP_list.append(AP.mean())
        AP_list.append([AP[i] for i in range(len(AP))])
        idx += 1

    # Best model
    print('-'*30)
    model_index = mAP_list.index(max(mAP_list))
    model_info = AP_list[model_index]
    print('maximum mAP: %f --> %d epoch'%(max(mAP_list), model_index))
    for i, c in enumerate(model_info):
        print("+ Class \'%d\' (%s) - AP: %f"%(i, class_names[i], model_info[i]))



    plt.title('mAP for each epoch(max: %f in %d epoch)'%(max(mAP_list), model_index))
    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.xlim(0, 100)
    plt.plot(mAP_list, c='k',marker='o', ls='--', lw=1, mfc='r', ms=3)
    plt.grid(True)
    plt.show()