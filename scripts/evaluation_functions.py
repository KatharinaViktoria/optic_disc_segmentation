"""
utility functions to evaluate segmentation 
and compute uncertainty metrics
most recent update: 11/01/2021 (data loading, Dice and Hausdorff)
"""

import numpy as np
import os

from metrics import compute_robust_hausdorff, compute_surface_distances
from PIL import Image
from scipy import ndimage



"""
# loader and saver functions
"""

def load_gt(data_dir, case_name, mode=None):

    gt = np.asarray(Image.open(os.path.join(data_dir, case_name+".bmp"))).astype(int)
    
    if mode == "cup":
        cup = np.zeros(gt.shape[:2])
        cup[np.where(gt[:,:,0] == 0)] = 1
        return cup
        
    elif mode == "disc":
        print("disc")
        disc = gt[:,:,1]
        return disc
    
    else:
        print("full gt")
        return gt


def load_pred(data_dir, case_name):
    
    return np.load(os.path.join(data_dir, case_name+".npy"))




"""
postprocessing functions 
"""

def binarize_pred(pred, binarization_threshold=.5):
    
    binarized = np.zeros(pred.shape)
    binarized[np.where(pred >= binarization_threshold)] = 1
    
    return binarized


def generate_mean_pred(pseudoprob_pred):

    return np.mean(pseudoprob_pred, axis=2)


def generate_max_softmax_uncertainty(pseudoprob_pred):

    return abs(pseudoprob_pred - 0.5)*(-2)+1 #absolute distance from .5 (and scale)


def entropy_map(pseudoprob):
    # voxelwise uncertainty map for foreground class

    return -np.multiply(pseudoprob,np.log(pseudoprob))
    
    
def generate_voxelwise_uncertainty(pred_pseudoprob): # generate uncertainty (entropy) map
    # average over voxelwise entropy of all predictions (compare comuputation of mean prediction)
    entropy_array = np.zeros(pred_pseudoprob.shape)

    for pred_number in range(pred_pseudoprob.shape[-1]):
        entropy_array[:,:,pred_number] = entropy_map(pred_pseudoprob[:, :, pred_number])

    return np.mean(entropy_array, axis=2) # = voxelwise uncertainty for structure s




"""
evaluation functions 
"""

def calc_dice(gt, pred, binarize=False):
    
    if binarize:
        pred = binarize_pred(pred)
        
    right_positives = np.round(np.multiply(pred,gt)).astype(int)
    dice = (2.0*np.sum(right_positives)+1)/(np.sum(pred)+np.sum(gt)+1)
    
    return dice

def calc_mean_voxelwise_uncertainty(voxelwise_uncertainty, pred, binarize=False, binarization_threshold=.5):
    # input: voxelwise uncertainty map and prediction (binarized or mean pseudoprob)
    # output: mean uncertainty over the labelled region

    if binarize:
        pred = binarize_pred(pred, binarization_threshold=binarization_threshold)
    else:
        pred = pred.astype(int) # just to make sure that the prediction is an integer array
    
    return np.mean(voxelwise_uncertainty[np.nonzero(pred)]) # calculate voxelwise uncertainty for all voxels labelled as foreground (mask and calc mean over non-zero values)


def calc_mean_pairwise_dice(pred_pseudoprob, binarization_threshold=.5):
  
    dice_list = []

    nr_samples = pred_pseudoprob.shape[-1]
    for i in range(nr_samples):
        prediction_1 = binarize_pred(pred_pseudoprob[:,:,i], binarization_threshold=binarization_threshold) # load and binarize pred i
        for j in range(i+1, nr_samples):
            prediction_2 = binarize_pred(pred_pseudoprob[:, :, j], binarization_threshold=binarization_threshold) # load and binarize pred j
            dice_list.append(calc_dice(prediction_1, prediction_2, binarize=False)) # calc dice and add to dice_list

    return np.mean(dice_list) # mean pairwise dice




"""
wrapper
"""

def compute_dice_dataset(names, gt_dir, pred_dir, mode=None, sampling=False): 

    # does not support multi-label eval yet
    dice_dict = dict()

    for n in names:
        
        gt = load_gt(gt_dir, n, mode=mode) # load gt
        pred = load_pred(pred_dir, n) # load pred
        if sampling:
            pred = generate_mean_pred(pred)

        dice = calc_dice(gt, pred, binarize=True)# compute_dice

        dice_dict[n] = dice

    return dice_dict


def compute_max_softmax_uncertainty_dataset(names, pred_dir, sampling=False): 
    # does not support multi-label prediction yet
    # somewhat depricated...

    uncertainty_dict = dict()

    for n in names:
        
        pred = load_pred(pred_dir, n) # load pred
        if sampling:
            pred = generate_mean_pred(pred)

        voxelwise_uncertainty = generate_max_softmax_uncertainty(pred) # create max softmax uncertainty map
        
        max_softmax_uncertainty = calc_mean_voxelwise_uncertainty(voxelwise_uncertainty, pred, binarize=True, binarization_threshold=.5) # compute uncertainty 

        uncertainty_dict[n] = max_softmax_uncertainty

    return uncertainty_dict


def compute_mean_uncertainty_dataset(names, pred_dir, sampling=False): 
    # does not support multi-label prediction yet

    uncertainty_dict = dict()

    for n in names:
        
        pred = load_pred(pred_dir, n) # load pred
        if sampling:
            voxelwise_uncertainty = generate_voxelwise_uncertainty(pred) # create max softmax uncertainty map
            pred = generate_mean_pred(pred)

        else:
            voxelwise_uncertainty = generate_max_softmax_uncertainty(pred) # create max softmax uncertainty map
        
        mean_uncertainty = calc_mean_voxelwise_uncertainty(voxelwise_uncertainty, pred, binarize=True, binarization_threshold=.5) # compute uncertainty 

        uncertainty_dict[n] = mean_uncertainty

    return uncertainty_dict


def compute_mean_pw_dice_dataset(names, pred_dir): 
    # does not support multi-label prediction yet

    pw_dice_dict = dict()

    for n in names:
        pred = load_pred(pred_dir, n) # load pred
        mean_pw_dice = calc_mean_pairwise_dice(pred, binarization_threshold=.5) # compute mean pairwise Dice  
        pw_dice_dict[n] = mean_pw_dice

    return pw_dice_dict