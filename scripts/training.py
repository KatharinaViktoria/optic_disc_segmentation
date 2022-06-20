"""
train optic disc and optic cup segmenter 
simple preprocessing

"""

import sys
sys.path.append("/kvh4/optic_disk/scripts") 

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random

from PIL import Image

from monai.data import Dataset, list_data_collate, PNGSaver # , decollate_batch
# from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.transforms import (
    LoadImage,
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate,
    RandSpatialCrop,
    ScaleIntensity,
    AsChannelFirst,
    AsChannelLast, 
    RandFlip,
    ToTensor,
    Resize,
)
from monai.visualize import plot_2d_or_3d_image
# from monai.data import ArrayDataset, create_test_image_2d # , decollate_batch
from torchvision.transforms import Lambda

from monai.utils import set_determinism
from monai.utils.misc import first
import torch

from typing import Optional, Callable

from dataset import ArrayDataset
from dropout_unet import DRUNet

"""
parameters: 
- number of training and validation samples (need to save (pickle) and  read in the dictionaries with the paths to the files) 
    -> test: the total number cannot be higher than the total number of files available
- network parameters  
- batch size, number of epochs 
- model directory 
- seed
"""
# files is a list: [train_images, train_gt, val_images, val_gt]


def seed_everything(seed=1234):
    # source: https://github.com/pytorch/pytorch/issues/11278
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_dice(seed, training_images, training_gt, validation_images, validation_gt, epoch_num, model_dir, p_dropout=0, learning_rate=1e-4):

    seed_everything(seed=seed)

    """
    data transformations and dataloader
    """
    # define training transformations (separately for input and gt)
    training_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    training_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )


    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    training_ds = ArrayDataset(training_images, training_imtransforms, training_gt, training_gttransforms)
    training_loader = torch.utils.data.DataLoader(training_ds, batch_size=12, shuffle=True)

    validation_ds = ArrayDataset(validation_images, validation_imtransforms, validation_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)

    """
    training set-up
    """

    device = torch.device("cuda:3")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = UNet(
            dimensions=2,
            in_channels=3,
            out_channels=1,
            channels=(8,16, 32, 64, 128),
            strides=(2, 2, 2, 2,2),
            # num_res_units=2,
            norm="batch",
            dropout = p_dropout
        ).to(device)

    
    dice_metric = DiceMetric(include_background=False, reduction="mean") #, get_not_nans=False)
    
    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    """
    training
    """
    val_interval = 1
    epoch_loss_values = list()
    metric_values = list()
    val_loss_values = list()
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in training_loader:
            step += 1
            inputs, labels = batch_data[0][0].to(device), batch_data[0][1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_len = len(training_ds) // training_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        savepath = os.path.join(model_dir, "epoch_"+str(epoch+1)+".pth")
        print("savepath: ", savepath)
        torch.save(model.state_dict(), savepath)
        print("saved model")
        
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                metric_sum = 0.0
                metric_count = 0
                epoch_val_loss = list()
                for val_data in validation_loader:
                    val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    epoch_val_loss.append(loss_function(val_outputs, val_labels).item())
                    # val_outputs = [post_trans_1(i) for i in val_outputs]
                    val_outputs = post_trans_1(val_outputs[0])
                    # compute metric for current iteration
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                val_loss_values.append(np.mean(epoch_val_loss))
                
                
                print(
                    "current epoch: {} current mean dice: {:.4f}".format(
                        epoch + 1, metric
                    )
                )
                print("val loss: ",np.mean(epoch_val_loss))
                
    np.save(os.path.join(model_dir, "epoch_loss.npy"), epoch_loss_values)
    np.save(os.path.join(model_dir, "val_metrics.npy"), metric_values)
    np.save(os.path.join(model_dir, "val_loss.npy"), val_loss_values)
        
def train_bce(seed, training_images, training_gt, validation_images, validation_gt, epoch_num, model_dir, p_weight, p_dropout=0, learning_rate=1e-4):
    # train UNet using binary cross entropy loss (with weight p_weight on the positive class)
    seed_everything(seed=seed)

    """
    data transformations and dataloader
    """
    # define training transformations (separately for input and gt)
    training_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    training_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )


    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    training_ds = ArrayDataset(training_images, training_imtransforms, training_gt, training_gttransforms)
    training_loader = torch.utils.data.DataLoader(training_ds, batch_size=12, shuffle=True)

    validation_ds = ArrayDataset(validation_images, validation_imtransforms, validation_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)

    """
    training set-up
    """

    device = torch.device("cuda:3")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = UNet(
            dimensions=2,
            in_channels=3,
            out_channels=1,
            channels=(8,16, 32, 64, 128),
            strides=(2, 2, 2, 2,2),
            # num_res_units=2,
            norm="batch",
            dropout = p_dropout
        ).to(device)

    
    dice_metric = DiceMetric(include_background=False, reduction="mean") #, get_not_nans=False)
    
    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output

    pos_weight = [p_weight]                                                           
    pos_weight = torch.Tensor(pos_weight).cuda(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    """
    training
    """
    val_interval = 1
    epoch_loss_values = list()
    metric_values = list()
    val_loss_values = list()
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in training_loader:
            step += 1
            inputs, labels = batch_data[0][0].to(device), batch_data[0][1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_len = len(training_ds) // training_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        savepath = os.path.join(model_dir, "epoch_"+str(epoch+1)+".pth")
        print("savepath: ", savepath)
        torch.save(model.state_dict(), savepath)
        print("saved model")
        
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                metric_sum = 0.0
                metric_count = 0
                epoch_val_loss = list()
                for val_data in validation_loader:
                    val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    epoch_val_loss.append(loss_function(val_outputs, val_labels.float()).item())
                    # val_outputs = [post_trans_1(i) for i in val_outputs]
                    val_outputs = post_trans_1(val_outputs[0])
                    # compute metric for current iteration
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                val_loss_values.append(np.mean(epoch_val_loss))
                
                
                print(
                    "current epoch: {} current mean dice: {:.4f}".format(
                        epoch + 1, metric
                    )
                )
                print("val loss: ",np.mean(epoch_val_loss))
                
    np.save(os.path.join(model_dir, "epoch_loss.npy"), epoch_loss_values)
    np.save(os.path.join(model_dir, "val_metrics.npy"), metric_values)
    np.save(os.path.join(model_dir, "val_loss.npy"), val_loss_values)
    

def train_bce_MC(seed, training_images, training_gt, validation_images, validation_gt, epoch_num, model_dir, p_dropout=0, pos_weight=10, learning_rate=1e-4):

    seed_everything(seed=seed)

    """
    data transformations and dataloader
    """
    # define training transformations (separately for input and gt)
    training_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    training_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )


    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    training_ds = ArrayDataset(training_images, training_imtransforms, training_gt, training_gttransforms)
    training_loader = torch.utils.data.DataLoader(training_ds, batch_size=12, shuffle=True)

    validation_ds = ArrayDataset(validation_images, validation_imtransforms, validation_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)

    """
    training set-up
    """

    device = torch.device("cuda:3")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = DRUNet(
            dimensions=2,
            in_channels=3,
            out_channels=1,
            channels=(8,16, 32, 64, 128),
            strides=(2, 2, 2, 2,2),
            # num_res_units=2,
            norm="batch",
            dropout = p_dropout
        ).to(device)

    
    dice_metric = DiceMetric(include_background=False, reduction="mean") #, get_not_nans=False)
    
    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output
    pos_weight = [pos_weight]                                                           
    pos_weight = torch.Tensor(pos_weight).cuda(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    """
    training
    """
    val_interval = 1
    epoch_loss_values = list()
    metric_values = list()
    val_loss_values = list()
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in training_loader:
            step += 1
            inputs, labels = batch_data[0][0].to(device), batch_data[0][1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_len = len(training_ds) // training_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        savepath = os.path.join(model_dir, "epoch_"+str(epoch+1)+".pth")
        print("savepath: ", savepath)
        torch.save(model.state_dict(), savepath)
        print("saved model")
        
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                metric_sum = 0.0
                metric_count = 0
                epoch_val_loss = list()
                for val_data in validation_loader:
                    val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
                    # roi_size = (96, 96)
                    # sw_batch_size = 4
                    # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = model(val_images)
                    epoch_val_loss.append(loss_function(val_outputs, val_labels.float()).item())
                    # val_outputs = [post_trans_1(i) for i in val_outputs]
                    val_outputs = post_trans_1(val_outputs[0])
                    # compute metric for current iteration
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                val_loss_values.append(np.mean(epoch_val_loss))
                
                
                print(
                    "current epoch: {} current mean dice: {:.4f}".format(
                        epoch + 1, metric
                    )
                )
                print("val loss: ",np.mean(epoch_val_loss))
                
    np.save(os.path.join(model_dir, "epoch_loss.npy"), epoch_loss_values)
    np.save(os.path.join(model_dir, "val_metrics.npy"), metric_values)
    np.save(os.path.join(model_dir, "val_loss.npy"), val_loss_values)

def train_dice_MC(seed, training_images, training_gt, validation_images, validation_gt, epoch_num, model_dir, p_dropout=0, learning_rate=1e-4):

    seed_everything(seed=seed)

    """
    data transformations and dataloader
    """
    # define training transformations (separately for input and gt)
    training_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    training_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            RandFlip(spatial_axis=1, prob=.5),
            RandFlip(spatial_axis=0, prob=.5),
            RandRotate(range_x=15, prob=0.3, keep_size=True),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )


    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    training_ds = ArrayDataset(training_images, training_imtransforms, training_gt, training_gttransforms)
    training_loader = torch.utils.data.DataLoader(training_ds, batch_size=12, shuffle=True)

    validation_ds = ArrayDataset(validation_images, validation_imtransforms, validation_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)

    """
    training set-up
    """

    device = torch.device("cuda:3")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = DRUNet(
            dimensions=2,
            in_channels=3,
            out_channels=1,
            channels=(8,16, 32, 64, 128),
            strides=(2, 2, 2, 2,2),
            # num_res_units=2,
            norm="batch",
            dropout = p_dropout
        ).to(device)

    
    dice_metric = DiceMetric(include_background=False, reduction="mean") #, get_not_nans=False)
    
    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    """
    training
    """
    val_interval = 1
    epoch_loss_values = list()
    metric_values = list()
    val_loss_values = list()
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in training_loader:
            step += 1
            inputs, labels = batch_data[0][0].to(device), batch_data[0][1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_len = len(training_ds) // training_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        savepath = os.path.join(model_dir, "epoch_"+str(epoch+1)+".pth")
        print("savepath: ", savepath)
        torch.save(model.state_dict(), savepath)
        print("saved model")
        
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                metric_sum = 0.0
                metric_count = 0
                epoch_val_loss = list()
                for val_data in validation_loader:
                    val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
                    # roi_size = (96, 96)
                    # sw_batch_size = 4
                    # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = model(val_images)
                    epoch_val_loss.append(loss_function(val_outputs, val_labels).item())
                    # val_outputs = [post_trans_1(i) for i in val_outputs]
                    val_outputs = post_trans_1(val_outputs[0])
                    # compute metric for current iteration
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                val_loss_values.append(np.mean(epoch_val_loss))
                
                
                print(
                    "current epoch: {} current mean dice: {:.4f}".format(
                        epoch + 1, metric
                    )
                )
                print("val loss: ",np.mean(epoch_val_loss))
                
    np.save(os.path.join(model_dir, "epoch_loss.npy"), epoch_loss_values)
    np.save(os.path.join(model_dir, "val_metrics.npy"), metric_values)
    np.save(os.path.join(model_dir, "val_loss.npy"), val_loss_values)


def run_inference(model_dir, model_epoch, test_images, test_gt, save_dir):
    # inference function for non MC models
    
    """
    Data
    """
    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    validation_ds = ArrayDataset(test_images, validation_imtransforms, test_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)


    """
    Inference set-up
    """
    device = torch.device("cuda:3")

    model = UNet(
                dimensions=2,
                in_channels=3,
                out_channels=1,
                channels=(8,16, 32, 64, 128),
                strides=(2, 2, 2, 2,2),
                # num_res_units=2,
                norm="batch",
                dropout = 0.0
            ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir,"epoch_"+str(model_epoch)+".pth")))
    model.eval()

    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output

    # save_dir = os.path.join(model_dir, "test_preds_epoch_"+str(model_epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    """
    iterative inference (save output to model dir)
    """
    with torch.no_grad():
        saver = PNGSaver(output_dir=save_dir)
        for val_data in validation_loader:
            val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = post_trans_1(val_outputs[0])
            
            # print(val_data[1][0])
            output_np = val_outputs.cpu().detach().numpy()[0,0,:,:]
            np.save(os.path.join(save_dir,val_data[1][0]+".npy"),output_np)

    print("finished inference")



def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def run_MC_inference(model_dir, model_epoch, test_images, test_gt, save_dir, iterations=10, dropout=.2):
    # inference function for MC models
    
    """
    Data
    """
    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    validation_ds = ArrayDataset(test_images, validation_imtransforms, test_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)


    """
    Inference set-up
    """
    device = torch.device("cuda:3")

    model = DRUNet(
                dimensions=2,
                in_channels=3,
                out_channels=1,
                channels=(8,16, 32, 64, 128),
                strides=(2, 2, 2, 2,2),
                # num_res_units=2,
                norm="batch",
                dropout = dropout
            ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir,"epoch_"+str(model_epoch)+".pth")))
    model.eval()
    enable_dropout(model)
    
    

    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output

    # save_dir = os.path.join(model_dir, "test_preds_epoch_"+str(model_epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize dictionary with empty lists for each case in the dataset
    pred_dict = dict()
    cases = [c[-9:-4] for c in test_images]
    for c in cases:
        pred_dict[c] = []

    """
    # iterative inference (save output to model dir)
    """
    with torch.no_grad():
        for i in range(iterations):
            for val_data in validation_loader:
                val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
                # define sliding window size and batch size for windows inference
                roi_size = (96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                # val_outputs = model(val_images)
                val_outputs = post_trans_1(val_outputs[0])

                output_np = val_outputs.cpu().detach().numpy()[0,0,:,:]
                pred_dict[val_data[1][0]].append(output_np)
                
    # convert the lists of predictions for each case to a numpy array     
    for case in cases:
        to_save = np.zeros((512,512,iterations))
        case_pred = pred_dict[case]
        for it in range(iterations):
            to_save[:,:,it] = case_pred[it]
        np.save(os.path.join(save_dir,case+".npy"),to_save)    

    print("finished inference")



def run_ensemble_inference(ensemble_dir, model_name_list, test_images, test_gt, save_dir):
    # inference function for MC models
    
    """
    Data
    """
    validation_imtransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor()
            ]
        )

    validation_gttransforms = Compose(
            [ LoadImage(image_only=True),
            AsChannelFirst(),
            ToTensor(),
            Lambda(lambda x: x[0,:,:]!=1),
            AddChannel()
            ]
        )

    validation_ds = ArrayDataset(test_images, validation_imtransforms, test_gt, validation_gttransforms)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=1, shuffle=False)


    """
    Inference set-up
    """
    device = torch.device("cuda:3")

    model = UNet(
                dimensions=2,
                in_channels=3,
                out_channels=1,
                channels=(8,16, 32, 64, 128),
                strides=(2, 2, 2, 2,2),
                # num_res_units=2,
                norm="batch",
                dropout = 0
            ).to(device)
    
    post_trans_1 = Compose([ AddChannel(), Activations(sigmoid=True)]) # transforms for the output
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize dictionary with empty lists for each case in the dataset
    pred_dict = dict()
    cases = [c[-9:-4] for c in test_images]
    for c in cases:
        pred_dict[c] = []

    """
    # iterative inference (save output to model dir)
    """
    for model_name in model_name_list:

        # load new model
        model.load_state_dict(torch.load(os.path.join(ensemble_dir, model_name)))
        model.eval()
        
        with torch.no_grad():
        
            for val_data in validation_loader:
                val_images, val_labels = val_data[0][0].to(device), val_data[0][1].to(device)
                # define sliding window size and batch size for windows inference
                roi_size = (96, 96)
                sw_batch_size = 16
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = post_trans_1(val_outputs[0])

                output_np = val_outputs.cpu().detach().numpy()[0,0,:,:]
                pred_dict[val_data[1][0]].append(output_np)
                    
    # convert the lists of predictions for each case to a numpy array  
    iterations = len(model_name_list)
    for case in cases:
        to_save = np.zeros((512,512,iterations))
        case_pred = pred_dict[case]
        for it in range(iterations):
            to_save[:,:,it] = case_pred[it]
        np.save(os.path.join(save_dir,case+".npy"),to_save)    

    print("finished inference")
