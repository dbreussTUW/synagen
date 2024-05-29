# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:49:13 2022

@author: david
"""

import torch
import os
import numpy as np
from PIL import Image
from datetime import datetime
from copy import deepcopy
import cv2
from PIL import Image
from DefaultDataset import DefaultDataset
import time
from shapeGenerator import randomShape
import yaml
import argparse

x = datetime.now()
date_day_month = x.strftime("%d-%m_%H-%M")

# =============================================================================
#           # Parsing arguments (configuration file, data paths)
# =============================================================================
parser = argparse.ArgumentParser(description="SYNAGEN: Synthetic image Anomaly GENeration tool")
parser.add_argument("configfile", help = "YAML file containing SYNAGEN parameters")
parser.add_argument("dataset", help = "Dataset path")
parser.add_argument("outputpath", help = "Output path for VADAR output")

parser.add_argument("--textures", help = "Texture dataset path")
parser.add_argument("--masked", default=False, action="store_true", help = "Using masks for allowed anomalous regions")

args = parser.parse_args()
configPath = args.configfile
datasetPath = args.dataset
outputpath = args.outputpath
texturesPath = args.textures
masks = args.masked

# =============================================================================
#           # Reading SYNAGEN parameters from configuration file
# =============================================================================
with open(configPath, 'r') as file:
    service = yaml.safe_load(file)

shapes               = service["meta"]["shapes"]
pixelManipulations   = service["meta"]["pixel_manipulations"]
multiplier           = service["meta"]["multiples"]
gaussianKernelFactor = service["shape"]["gaussianKernelFactor"]
gaussianSigmaFactor  = service["shape"]["gaussianSigmaFactor"]
baseVal              = service["shape"]["baseVal"]
threshold            = service["shape"]["threshold"]
brighnessFactors     = service["texture"]["brighnessFactors"]
smoothingKernel      = service["texture"]["smoothingKernel"]
smoothingSigma       = service["texture"]["smoothingSigma"]

print(texturesPath)
print(pixelManipulations)
textures = False
if texturesPath is not None and "given-texture" in pixelManipulations:
    textures = True
    print("With texture: {}".format(texturesPath))
elif texturesPath is None and "given-texture" in pixelManipulations:
    pixelManipulations.remove("given-texture")
    print("No texture: {}".format(texturesPath))

if masks is None:
    masks = False

# =============================================================================
#           # Preparing dataloaders for dataset
# =============================================================================
dataset = DefaultDataset(datasetPath)
dataset_len = len(dataset)
data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=1, shuffle=False)
print("length of dataset = {}".format(len(dataset)))
if textures:
    tex_ds = DefaultDataset(texturesPath)
    tex_loader = torch.utils.data.DataLoader(dataset = tex_ds, batch_size=1, shuffle=True)
    iter_tex_loader = iter(tex_loader)
    tex_ds_len = len(tex_ds)
    print("length of textures-dataset = {}".format(len(tex_ds)))

# absolute size edge definition of anomaly size distribution
size_edges = service["meta"]["size_edges"]
for sample in data_loader:
    bs,ch,rows,cols = sample["image"].shape
    break
size_edges = [i*rows*cols for i in size_edges]

i = 0
# ===========================================================================
#       # Generation Loop: Every combination of the defined shapes,
#       # pixel manipulations, and sizes is applied on each image "multiples"
#       # times (defined in the configuration yaml file).
# ===========================================================================
for shape in shapes:
    for pixelManipulation in pixelManipulations:
        # Definition of the path for saving the artificial anomaly images
        path = outputpath + "/artificial_test/{}_{}".format(pixelManipulation, shape)
        exist = os.path.exists(path)
        if not exist:
            try:
                os.makedirs(path)
            except:
                print("failed to make dir {}".format(path))
        # Definition of the path for saving the ground truth images of each artificial anomaly image
        absdiff_path = outputpath + "/artificial_ground_truth/{}_{}".format(pixelManipulation, shape)
        exist = os.path.exists(absdiff_path)
        if not exist:
            try:
                os.makedirs(absdiff_path)
            except:
                print("failed to make dir {}".format(absdiff_path))
        cum_loss = 0.0
        j = 0
        file = open(path+"/parameterLog_{}_{}_{}.csv".format(pixelManipulation, shape, date_day_month),"w")
        file.write("FrameID, ImageName, baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, centers, gaussianKernelFactor, gaussianSigmaFactor, baseVal, threshold, R, G, B, anomaly mask size\n")
                
        for sample in data_loader:
            multiples = 0
            j = j + 1
            # create multiple anomalous images of each anomaly type from the same image
            while multiples < multiplier:
            
                multiples += 1
                loopStart = time.time()
                frameID = sample['imageName']
                img = sample['image']
                b,ch,ri,ci = img.shape
                img_view = img.view((-1,3,ri,ci))
                img_np = img_view.detach().cpu().numpy()

                # =============================================================================
                #           # Random shape generation
                # =============================================================================
                baseDim              = np.random.randint(service["shape"]["baseDim"][0], service["shape"]["baseDim"][1])
                sizeFactor           = np.random.uniform(service["shape"]["sizeFactor"][0], service["shape"]["sizeFactor"][1])
                innerWidthFactor     = np.random.uniform(service["shape"]["innerWidthFactor"][0], service["shape"]["innerWidthFactor"][1])
                innerHeightFactor    = service["shape"]["innerWidthFactor"][0]*service["shape"]["innerWidthFactor"][1]/innerWidthFactor
                outerCircleFactor    = np.random.uniform(service["shape"]["outerCircleFactor"][0],service["shape"]["outerCircleFactor"][1])
                centers              = np.random.randint(5,15)

                if "rough" in shape:
                    baseDim=np.random.randint(600,900)
            
                anomalyShapeOriginal = randomShape(shape, baseDim, sizeFactor,innerWidthFactor,innerHeightFactor,outerCircleFactor,centers,gaussianKernelFactor,gaussianSigmaFactor,baseVal,threshold)
                original_rows, original_cols = anomalyShapeOriginal.shape
                # making sure generated anomaly is not too small
                while np.sum(anomalyShapeOriginal) < size_edges[0]:
                    anomalyShapeOriginal = randomShape(shape, baseDim, sizeFactor,innerWidthFactor,innerHeightFactor,outerCircleFactor,centers,gaussianKernelFactor,gaussianSigmaFactor,baseVal,threshold)                                    
                actual_size = np.sum(anomalyShapeOriginal)

                # =============================================================================
                #             # resize shapes
                # =============================================================================
                sizes = [-1]*(len(size_edges)-1)
                factors = [1]*(len(size_edges)-1)
                
                for idx, x in enumerate(size_edges):
                    if idx < (len(size_edges)-1):
                        sizes[idx] = np.random.randint(x, size_edges[idx+1])
                        factors[idx] = sizes[idx]/actual_size

                f = 0
                
                for factor in factors:
                    # determine new shape
                    target_size = int(actual_size*factor)
                    new_cols = int(original_cols*np.sqrt(factor))
                    new_rows = int(original_rows*np.sqrt(factor))
                    
                    # make sure new dimensions are valid
                    if new_cols >= ci or new_rows >= ri:
                        if new_cols >= ci:
                            new_cols = ci-1
                            new_rows = original_rows*factor/(new_cols/original_cols)
                            if new_rows >= ri:
                                new_rows = ri-1
     
                        else:
                            new_rows = ri-1
                            new_cols = original_cols*factor/(new_rows/original_rows)
                            if new_cols >= ci:
                                new_cols = ci-1
                    
                    newSize = cv2.resize(anomalyShapeOriginal, dsize=(new_cols, new_rows), interpolation=cv2.INTER_AREA)
        
                    dim = np.max(newSize.shape) 
                    temp_newSize = np.zeros((dim*3,dim*3))
                    temp_newSize[dim:dim+newSize.shape[0], dim:dim+newSize.shape[1]] = newSize
                    
                    # =============================================================================
                    #       # rotate shape by random angle
                    # =============================================================================
                    angle = np.random.uniform(0,359)
                    rs = Image.fromarray(temp_newSize).rotate(angle)
                    rs = np.asarray(rs)
                    
                    if rs.shape[0] < ri and rs.shape[1] < ci:
                        rs_th = np.zeros(rs.shape)
                        rs_th[rs>=0.5] = 1.0
                        indices = np.argwhere(rs_th > 0.5)
                        
                        # in case shrinking fails
                        if len(indices) < 1:
                            newSize_crop = randomShape(shape, 100, sizeFactor,innerWidthFactor,innerHeightFactor,outerCircleFactor,centers,gaussianKernelFactor,gaussianSigmaFactor,baseVal,threshold)
                        else:
                            newSize_crop = np.zeros(rs.shape)
                            newSize_crop = rs_th[np.min(indices[:,0]):np.max(indices[:,0]), np.min(indices[:,1]):np.max(indices[:,1])]
                    else:
                        rs_th = np.zeros(rs.shape)
                        rs_th[rs>=0.5] = 1.0
                        indices = np.argwhere(rs_th > 0.5)
                        newSize_temp = rs_th[np.min(indices[:,0]):np.max(indices[:,0]), np.min(indices[:,1]):np.max(indices[:,1])]
                        newSize_crop = np.zeros((ri-10, ci-10))
                        newSize_crop = newSize_temp[5:ri-5, 5:ci-5]
                    
                    anomalyShape = newSize_crop
                    rows, cols = anomalyShape.shape
                    
                    ###################################################################
                    
                    temp = np.uint8(255*anomalyShape)
                    ret, thresh = cv2.threshold(temp, 122, 255, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                    xc = np.uint8(np.zeros((rows,cols)))
                    cv2.drawContours(xc, contours, contourIdx=-1, color=255, thickness=3)
                    xw = np.zeros((3,ri,ci))
                    
                    singleChannelAnomalyPlane = np.zeros((ri,ci))
                    
                    anSh = np.zeros((1,3,rows,cols))
                    anSh[0,0,:,:] = anomalyShape
                    anSh[0,1,:,:] = anomalyShape
                    anSh[0,2,:,:] = anomalyShape
                    
                    # =============================================================================
                    #     # random position
                    # =============================================================================
                    if masks:
                        try:
                            mask = Image.open(sample['imagePath'][0].replace(".", "_mask."))
                            mask = np.asarray(mask)
                            mask = mask/255.0
                        except:
                            print("Mask file not valid! ({})".format(sample['imagePath'][0].replace(".", "_mask.")))
                            mask = np.ones((ri,ci,3))
                            mask[0:int(rows*0.5),:] = 0.0
                            mask[int(ri-rows*0.5):,:] = 0.0
                            mask[:,0:int(cols*0.5)] = 0.0
                            mask[:,int(ci-cols*0.5):] = 0.0
                        mask = cv2.resize(mask, dsize=(ri,ci))
                        # find regions where anomalies are allowed
                        mask_indices = np.argwhere(mask > 0)
                        ind_pair = np.random.randint(0,len(mask_indices))
                        x = mask_indices[ind_pair,1]-int(0.5*cols) if mask_indices[ind_pair,1]-int(0.5*cols) >= 0 else 0
                        y = mask_indices[ind_pair,0]-int(0.5*rows) if mask_indices[ind_pair,0]-int(0.5*rows) >= 0 else 0

                        mask = np.einsum("kij->jki", mask)[np.newaxis,...]
                        anomalyPlane = np.zeros((1,3,ri,ci))
                        # add anomaly at random allowed position
                        if y+rows >= ri and x+cols >= ci:
                            anomalyPlane[:,:,y:,x:] = anSh[:,:,:ri-y,:ci-x]
                        elif y+rows >= ri:
                            anomalyPlane[:,:,y:,x:x+cols] = anSh[:,:,:ri-y,:]
                        elif x+cols >= ci:
                            anomalyPlane[:,:,y:y+rows,x:] = anSh[:,:,:,:ci-x]
                        else:
                            anomalyPlane[:,:,y:y+rows,x:x+cols] = anSh
                        anomalyPlane = np.multiply(anomalyPlane, mask)
                        # view only anomalous region to determine contours
                        tempAnomalyPlane = anomalyPlane[0,0,:,:]
                        # update size since actual size is changed through the mask limitations
                        target_size = np.count_nonzero(tempAnomalyPlane)
                        try:
                            indices = np.argwhere(tempAnomalyPlane > 0)
                            row_min = np.min(indices[:,0])
                            row_max = np.max(indices[:,0])
                            col_min = np.min(indices[:,1])
                            col_max = np.max(indices[:,1])
                        except ValueError:
                            print("SYNAGEN struggles to find valid position for given anomaly shape. Example is skipped.")
                            continue
                        
                        rows = row_max-row_min
                        cols = col_max-col_min
                        anomalyShape = tempAnomalyPlane[row_min:row_max, col_min:col_max]
                        tempRegion = tempAnomalyPlane[row_min:row_max, col_min:col_max]
                        # calculate contours
                        temp = np.uint8(255*tempRegion)
                        ret, thresh = cv2.threshold(temp, 122, 255, cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                        xc = np.uint8(np.zeros((rows,cols)))
                        cv2.drawContours(xc, contours, contourIdx=-1, color=255, thickness=3)
                        xw = np.zeros((3,ri,ci))

                        xw[0,y:y+rows,x:x+cols] = xc
                        xw[1,y:y+rows,x:x+cols] = xc
                        xw[2,y:y+rows,x:x+cols] = xc
                    else:
                        x = np.random.randint(0,ci-cols)
                        y = np.random.randint(0,ri-rows)

                        anomalyPlane = np.zeros((1,3,ri,ci))
                        anomalyPlane[:,:,y:y+rows,x:x+cols] = anSh
                        xw[0,y:y+rows,x:x+cols] = xc
                        xw[1,y:y+rows,x:x+cols] = xc
                        xw[2,y:y+rows,x:x+cols] = xc           
                    
                    singleChannelAnomalyPlane[y:y+rows,x:x+cols] = anomalyShape
                    anomalyImg = deepcopy(img_np)


                    # =============================================================================
                    #     # R,G,B channel factors definition
                    # =============================================================================
                    if np.random.rand() > 0.5:
                        if np.random.rand() > 0.5:
                            r = np.random.uniform(brighnessFactors[0],brighnessFactors[1])
                        else:
                            r = np.random.uniform(brighnessFactors[2],brighnessFactors[3])
                        g = r
                        b = r
                        
                    else:
                        if np.random.rand() > 0.5:
                            r = np.random.uniform(brighnessFactors[0],brighnessFactors[1])
                        else:
                            r = np.random.uniform(brighnessFactors[2],brighnessFactors[3])
                            
                        if np.random.rand() > 0.5:
                            g = np.random.uniform(brighnessFactors[0],brighnessFactors[1])
                        else:
                            g = np.random.uniform(brighnessFactors[2],brighnessFactors[3])
                        
                        if np.random.rand() > 0.5:
                            b = np.random.uniform(brighnessFactors[0],brighnessFactors[1])
                        else:
                            b = np.random.uniform(brighnessFactors[2],brighnessFactors[3])
                    # =============================================================================
                    #     # applying pixel manipulation method
                    # =============================================================================
                
                    pixelManipulation = pixelManipulation    # "transparent", "tran-upscaled noise", "tran-col-upscaled noise", "gray-upscaled noise", "mean-upscaled noise"

                    if "transparent" in pixelManipulation:
                        # multiply each channel by an individual factor
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = anomalyImg[0,0,singleChannelAnomalyPlane>0.5]*r
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = anomalyImg[0,1,singleChannelAnomalyPlane>0.5]*g
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = anomalyImg[0,2,singleChannelAnomalyPlane>0.5]*b
                    elif "tran-upscaled noise" in pixelManipulation:
                        # multiply each channel by the same factor and add upscaled noise
                        startDim = np.random.randint(50, ri)
                        x = np.random.rand(startDim,startDim)*0.5 + 1
                        y = cv2.resize(x, dsize=(ri,ci))
                        gK = 11
                        gS = 7
                        cv2.GaussianBlur(y, (gK,gK), gS)
                        
                        # ensuring gray
                        if np.random.rand() > 0.5:
                            r = np.random.uniform(brighnessFactors[0],brighnessFactors[1])
                        else:
                            r = np.random.uniform(brighnessFactors[2],brighnessFactors[3])
                        g = r
                        b = r
                        
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = np.minimum(np.multiply(anomalyImg[0,0,singleChannelAnomalyPlane>0.5]*r, y[singleChannelAnomalyPlane>0.5]), np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = np.minimum(np.multiply(anomalyImg[0,1,singleChannelAnomalyPlane>0.5]*g, y[singleChannelAnomalyPlane>0.5]), np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = np.minimum(np.multiply(anomalyImg[0,2,singleChannelAnomalyPlane>0.5]*b, y[singleChannelAnomalyPlane>0.5]), np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                    elif "tran-col-upscaled noise" in pixelManipulation:
                        # multiply each channel by an individual factor and add upscaled noise
                        startDim = np.random.randint(50, ri)
                        x = np.random.rand(startDim,startDim)*0.5 + 1
                        y = cv2.resize(x, dsize=(ri,ci))
                        gK = 11
                        gS = 7
                        cv2.GaussianBlur(y, (gK,gK), gS)
                        
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = np.minimum(np.multiply(anomalyImg[0,0,singleChannelAnomalyPlane>0.5]*r, y[singleChannelAnomalyPlane>0.5]), np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = np.minimum(np.multiply(anomalyImg[0,1,singleChannelAnomalyPlane>0.5]*g, y[singleChannelAnomalyPlane>0.5]), np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = np.minimum(np.multiply(anomalyImg[0,2,singleChannelAnomalyPlane>0.5]*b, y[singleChannelAnomalyPlane>0.5]), np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                    elif "gray-upscaled noise" in pixelManipulation:
                        # replace pixel values with gray upscaled noise
                        startDim = np.random.randint(50, ri)
                        x = np.random.rand(startDim,startDim)
                        y = cv2.resize(x, dsize=(ri,ci))
                        gK = 11
                        gS = 7
                        cv2.GaussianBlur(y, (gK,gK), gS)
                        
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = y[singleChannelAnomalyPlane>0.5]
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = y[singleChannelAnomalyPlane>0.5]
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = y[singleChannelAnomalyPlane>0.5]
                        
                    elif "mean-upscaled noise" in pixelManipulation:
                        # replace pixel values with mean values and add upscaled noise
                        startDim = np.random.randint(50, ri)
                        x = np.random.rand(startDim,startDim)
                        y = cv2.resize(x, dsize=(ri,ci))
                        gK = 11
                        gS = 7
                        cv2.GaussianBlur(y, (gK,gK), gS)
                        
                        if np.random.rand() > 0.5:
                            factor = np.random.uniform(brighnessFactors[0],brighnessFactors[1])
                        else:
                            factor = np.random.uniform(brighnessFactors[2],brighnessFactors[3])
                        
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = np.minimum(y[singleChannelAnomalyPlane>0.5]*0.2+np.mean(anomalyImg[0,0,singleChannelAnomalyPlane>0.5])*factor, np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = np.minimum(y[singleChannelAnomalyPlane>0.5]*0.2+np.mean(anomalyImg[0,1,singleChannelAnomalyPlane>0.5])*factor, np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = np.minimum(y[singleChannelAnomalyPlane>0.5]*0.2+np.mean(anomalyImg[0,2,singleChannelAnomalyPlane>0.5])*factor, np.ones((ri,ci))[singleChannelAnomalyPlane>0.5])
                    
                    elif "given-texture" in pixelManipulation:
                        # replace pixel values with texture regions (images within "texturepath" directory)
                        startDim = np.random.randint(50, ri)
                        x = np.random.rand(startDim,startDim)
                        y = cv2.resize(x, dsize=(ri,ci))
                        gK = 11
                        gS = 7
                        cv2.GaussianBlur(y, (gK,gK), gS)
                        try:
                            texture = next(iter_tex_loader)["image"][0].detach().cpu().numpy()
                        except StopIteration:
                            iter_tex_loader = iter(tex_loader)
                            texture = next(iter_tex_loader)["image"][0].detach().cpu().numpy()
                        texture = np.einsum("kij->ijk", texture)
                        texture_resized = cv2.resize(texture, dsize=(ri,ci))
                        texture_resized = np.einsum("kij->jki", texture_resized)
                        
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = texture_resized[0,singleChannelAnomalyPlane>0.5]
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = texture_resized[1,singleChannelAnomalyPlane>0.5]
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = texture_resized[2,singleChannelAnomalyPlane>0.5]

                    else:
                        # default is transparent
                        anomalyImg[0,0,singleChannelAnomalyPlane>0.5] = anomalyImg[0,0,singleChannelAnomalyPlane>0.5]*r
                        anomalyImg[0,1,singleChannelAnomalyPlane>0.5] = anomalyImg[0,1,singleChannelAnomalyPlane>0.5]*g
                        anomalyImg[0,2,singleChannelAnomalyPlane>0.5] = anomalyImg[0,2,singleChannelAnomalyPlane>0.5]*b
                
                    # =============================================================================
                    #       # blurring the edges
                    # =============================================================================
                    tempImg = np.einsum("kij->ijk", deepcopy(anomalyImg[0]))
                    tempImg = cv2.GaussianBlur(tempImg, (smoothingKernel, smoothingKernel), smoothingSigma)
                    tempImg = np.einsum("kij->jki", tempImg)
                    blurredEdges = deepcopy(anomalyImg[0])
                    blurredEdges[xw>0] = tempImg[xw>0]
                    blurredEdges = np.einsum("kij->ijk", deepcopy(blurredEdges))
                    
                    loopEnd = time.time()                    
                    
                    # image postprocessing
                    blurredEdges[blurredEdges>=1.0] = 1.0
                    diff_img = np.abs(np.einsum('kij->ijk', img_np[0])-blurredEdges)
                    save_diff_img = 255*diff_img
                    save_diff_img = np.uint8(save_diff_img)
                    save_diff_img = Image.fromarray(save_diff_img)

                    save_img = 255*blurredEdges
                    save_img = np.uint8(save_img)
                    save_img = Image.fromarray(save_img)
                    
                    save_img.save(path + "/{}_{}_{}-{}-{}.png".format(pixelManipulation, shape, j, f, multiples))                    
                    save_diff_img.save(absdiff_path + "/{}_{}_{}-{}-{}.png".format(pixelManipulation, shape, j, f, multiples))

                    # logging parameters
                    name = "{}_{}_{}-{}-{}.png".format(pixelManipulation, shape, j, f, multiples)
                    file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(frameID, name, baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, centers, gaussianKernelFactor, gaussianSigmaFactor, baseVal, threshold, r,g,b, target_size))

                    f += 1
                
        file.close()