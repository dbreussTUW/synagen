# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:02:45 2023

@author: dbr
"""

import numpy as np
import cv2

def initializeParameters(baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, gaussianKernelFactor, gaussianSigmaFactor):
    # defining shape parameters
    size = int(baseDim*sizeFactor)
    innerWidth = int(baseDim*innerWidthFactor) if int(baseDim*innerWidthFactor) > 0 else 1
    innerHeight = int(baseDim*innerHeightFactor) if int(baseDim*innerHeightFactor) > 0 else 1
    outerCircle = int(baseDim*outerCircleFactor) if int(baseDim*outerCircleFactor) > 2 else 2
    gaussianKernel = int(baseDim*gaussianKernelFactor)
    # ensuring odd kernel size
    if gaussianKernel%2 == 0:
        gaussianKernel = gaussianKernel+1
    gaussianSigma = int(baseDim*gaussianSigmaFactor)
    
    # set initial range of valid first cluster center position
    lowx = int(0.4*baseDim)
    highx = int(0.6*baseDim)
    lowy = int(0.4*baseDim)
    highy = int(0.6*baseDim)

    return size, innerWidth, innerHeight, outerCircle, gaussianKernel, gaussianSigma, lowx, highx, lowy, highy

def clusterPlacement(centers, lowx, highx, lowy, highy, innerWidth, innerHeight, outerCircle, baseDim, size, baseVal):
    plane = np.ones((baseDim,baseDim))*baseVal
    # place cluster centers
    for i in range(centers):
        # randomly sampling center points
        x1 = np.random.randint(lowx, highx)
        y1 = np.random.randint(lowy, highy)
        
        # set valid x and y coordinates of cluster centers and boarder outlines
        xout0 = x1-outerCircle-innerWidth if x1-outerCircle-innerWidth >= 0 else 0
        xout2 = x1+outerCircle+innerWidth if x1+outerCircle+innerWidth < baseDim else baseDim-1
        yout0 = y1-outerCircle-innerHeight if y1-outerCircle-innerHeight >= 0 else 0
        yout2 = y1+outerCircle+innerHeight if y1+outerCircle+innerHeight < baseDim else baseDim-1
        
        xin0 = x1-innerWidth if x1-innerWidth >= 0 else 0
        xin2 = x1+innerWidth if x1+innerWidth < baseDim else baseDim-1
        yin0 = y1-innerHeight if y1-innerHeight >= 0 else 0
        yin2 = y1+innerHeight if y1+innerHeight < baseDim else baseDim-1
        
        # set pixels within boarder region to one baseVal and pixels within cluster center to 2*baseVal
        plane[yout0:yout2, xout0:xout2] = plane[yout0:yout2, xout0:xout2] + baseVal
        plane[yin0:yin2, xin0:xin2] = plane[yin0:yin2, xin0:xin2] + 2*baseVal
        
        # fix new range of valid cluster center position
        if i == 0:
            lowx = x1-size if x1-size >= 0 else 0
            highx = x1+size if x1+size < baseDim-1 else baseDim-1
            lowy = y1-size if y1-size >= 0 else 0
            highy = y1+size if y1+size < baseDim-1 else baseDim-1
        
    return plane

def smoothingPlane(baseDim, plane, gaussianKernel, gaussianSigma, threshold):
    # Multiplying plane pixels with random numbers between 0 and 1  
    randomPlane = np.random.random((baseDim,baseDim))
    mulPlane = np.multiply(plane, randomPlane)
    # Gaussian Blurring for smooting the shape
    blurredPlane = cv2.GaussianBlur(mulPlane, (gaussianKernel,gaussianKernel), gaussianSigma)
    thresholdedPlane = np.zeros((baseDim,baseDim))
    # Thresholding for final "Spattered" anomaly shape
    thresholdedPlane[blurredPlane > threshold] = 1.0
    return thresholdedPlane

def randomShape(style="spattered", baseDim=100, sizeFactor=0.3, innerWidthFactor=0.05, innerHeightFactor=0.05, outerCircleFactor=0.05, centers=10, gaussianKernelFactor=0.05, gaussianSigmaFactor=0.07, baseVal=0.2, threshold=0.3):
    
    if "spattered" in style:
        #============================================
        # "Spattered" shape
        #============================================
        # initialize parameters
        size, innerWidth, innerHeight, outerCircle, gaussianKernel, gaussianSigma, lowx, highx, lowy, highy = initializeParameters(\
            baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, gaussianKernelFactor, gaussianSigmaFactor)
        # place cluster centers
        plane = clusterPlacement(centers, lowx, highx, lowy, highy, innerWidth, innerHeight, outerCircle, baseDim, size, baseVal)
                
    elif "complex" in style:
        #=================================================================
        # "Complex" shape (same as "Spattered" but additional second step)
        #=================================================================
        # initialize parameters
        size, innerWidth, innerHeight, outerCircle, gaussianKernel, gaussianSigma, lowx, highx, lowy, highy = initializeParameters(\
            baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, gaussianKernelFactor, gaussianSigmaFactor)
        # place cluster centers
        plane = clusterPlacement(centers, lowx, highx, lowy, highy, innerWidth, innerHeight, outerCircle, baseDim, size, baseVal)
        # smoothing the plane
        thresholdedPlane_1 = smoothingPlane(baseDim, plane, gaussianKernel, gaussianSigma, threshold)
        
        #===============================================
        # Additional second step for the "Complex" shape
        #===============================================
        # Threshold to randomize sparseness of "complex" shape
        th = np.random.uniform(0.3, 0.6)
        # Create small scale noise of variable size by drawing size from uniform distribution
        startDim = np.random.randint(int(0.03*baseDim), int(baseDim*0.1))
        x = np.random.rand(startDim,startDim)
        # Upscaling random plane to same dimension as primary "Spattered" shape
        randPlane = cv2.resize(x, dsize=(baseDim,baseDim))
        randPlane[randPlane < th] = 0.0
        randPlane[randPlane >= th] = 1.0
        # Multiplying primary "Spattered" shape with random plane
        thresholdedPlaneTemp = np.multiply(thresholdedPlane_1, randPlane)
        # Making sure gaussian kernel is odd
        if gaussianKernel%2 == 0:
            gaussianKernel = gaussianKernel+1
        # Blurring
        blurredPlane = cv2.GaussianBlur(thresholdedPlaneTemp, (gaussianKernel,gaussianKernel), sigmaX=0.3, sigmaY=0.3)
        # Thresholding
        thresholdedPlane = np.zeros((baseDim,baseDim))
        thresholdedPlane[blurredPlane > threshold*1.5] = 1.0

        return thresholdedPlane
     
    elif "rough" in style:
        #=================================================================
        # "Rough" shape (same as "Spattered" but additional second step)
        #=================================================================
        # initialize parameters
        size, innerWidth, innerHeight, outerCircle, gaussianKernel, gaussianSigma, lowx, highx, lowy, highy = initializeParameters(\
            baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, gaussianKernelFactor, gaussianSigmaFactor)
        # place cluster centers
        plane = clusterPlacement(centers, lowx, highx, lowy, highy, innerWidth, innerHeight, outerCircle, baseDim, size, baseVal)
        # smoothing the plane
        thresholdedPlane = smoothingPlane(baseDim, plane, gaussianKernel, gaussianSigma, threshold)
        
        #===============================================
        # Additional second step for the "Rough" shape
        #===============================================
        # get minimum and maximum row and col indices
        indices = np.argwhere(thresholdedPlane > 0.5)
        row_indices = indices[:,0]
        col_indices = indices[:,1]
        if len(row_indices) <= 1:
            row_min = 0
            row_max = 2
            col_min = 0
            col_max = 2
        
        elif len(row_indices) <= 1:
            row_min = 0
            row_max = 2
            col_min = 0
            col_max = 2
        else:
            row_min = np.min(row_indices)
            row_max = np.max(row_indices)
            col_min = np.min(col_indices)
            col_max = np.max(col_indices)
        
        # Defining Scratch Plane with number of scratches drawn from uniform distribution
        scratchPlane = np.zeros((baseDim, baseDim))
        scratchNumber = np.random.randint(1,5)
        
        for i in range(scratchNumber):
            # define row and col start and end indices
            try:
                row_start = np.random.randint(row_min+2, row_min+0.4*(row_max-row_min)+2)
                row_end = np.random.randint(row_max-2-0.4*(row_max-row_min), row_max-2)
                col_start = np.random.randint(col_min+2, col_min+0.4*(col_max-col_min)+2)
                col_end = np.random.randint(col_max-2-0.4*(col_max-col_min), col_max-2)
            except:
                row_start = row_min
                row_end = row_max-1
                col_start = col_min
                col_end = col_max-1
            
            if (row_end-row_start) <= (col_end-col_start):
                # arange row vector with increments of one and evenly spaced according coloumn vector
                row_vals = np.arange(row_start, row_end, 1, dtype=np.int16)
                step_size = row_end-row_start
                step_size = np.maximum(1,step_size)
                if step_size<=0:
                    step_size=1
                col_vals = np.arange(col_start, col_end, (col_end-col_start)/step_size)
                col_vals = np.int16(col_vals)
            else:
                # arange coloumn vector with increments of one and evenly spaced according row vector
                col_vals = np.arange(col_start, col_end, 1, dtype=np.int16)
                step_size = col_end-col_start
                step_size = np.maximum(1,step_size)
                if step_size<=0:
                    step_size=1
                row_vals = np.arange(row_start, row_end, (row_end-row_start)/step_size)
                row_vals = np.int16(row_vals)
            
            if row_vals.shape == col_vals.shape:
                # print("ok,tiptop")
                row_vals = row_vals
            else:
                # make sure rows and coloumn vals are of same shape
                if row_vals.shape[0] > col_vals.shape[0]:
                    col_vals = np.resize(col_vals, row_vals.shape)
                else:
                    row_vals = np.resize(row_vals, col_vals.shape)
            
            # randomly flip rows
            if np.random.rand() > 0.5:
                row_vals = np.flip(row_vals)
            
            # draw scratches
            scratchPlane[row_vals,col_vals] = 1.0
            
            # ignore row and coloumn values right at the borders
            row_vals[row_vals<=0] = 1
            col_vals[col_vals<=0] = 1
            row_vals[row_vals==baseDim-1] = baseDim-2
            col_vals[col_vals==baseDim-1] = baseDim-2
            
            # widen scratches by 1 in each direction
            scratchPlane[row_vals-1, col_vals] = 1.0
            scratchPlane[row_vals, col_vals-1] = 1.0

            # add additional scratches with same orientation (parallel shifts), number drawn from uniform distribution
            multiples = np.random.randint(3,10)
            for i in range(multiples):
                try:
                    offset_row = np.random.randint(row_min, row_start)
                    offset_col = np.random.randint(col_min, col_start)
                except:
                    offset_row = row_min
                    offset_col = col_min
                
                if offset_row > np.min(row_vals):
                    offset_row = np.maximum(int(np.min(row_vals))-1,0)
                if offset_col > np.min(col_vals):
                    offset_col = np.maximum(int(np.min(col_vals))-1,0)
                
                scratchPlane[row_vals-offset_row,col_vals-offset_col] = 1.0
                scratchPlane[row_vals-offset_row-1, col_vals-offset_col] = 1.0
                scratchPlane[row_vals-offset_row, col_vals-offset_col-1] = 1.0

        # Salt noise plane, randomly sampled thresholdSalt defines sparsity of salt noise plane
        saltNoise = np.random.rand(baseDim, baseDim)
        thresholdSalt = np.random.uniform(0.7,0.95)
        saltNoise[saltNoise < thresholdSalt] = 0
        saltNoise[saltNoise > thresholdSalt] = 1.0
        
        # Adding salt noise to scratch plane and multiply with primary "Spattered" mask
        scratchPlane = np.multiply(np.maximum(saltNoise, scratchPlane), thresholdedPlane)
        # Gaussian Blurring for smooting the shape
        blurredPlane = cv2.GaussianBlur(scratchPlane, (3,3), 1)
        thresholdedPlane = np.zeros((baseDim,baseDim))
        # Thresholding for final "Spattered" anomaly shape
        thresholdedPlane[blurredPlane > threshold] = 1.0
 
        return thresholdedPlane

    elif "elongated" in style:
        #============================================
        # "Elongated" shape
        #============================================
        # initialize parameters
        baseVal = baseVal*1.5
        size, innerWidth, innerHeight, outerCircle, gaussianKernel, gaussianSigma, lowx, highx, lowy, highy = initializeParameters(\
            baseDim, sizeFactor, 4*innerWidthFactor, 0.25*innerHeightFactor, 0.25*outerCircleFactor, gaussianKernelFactor, gaussianSigmaFactor)
        plane = np.ones((baseDim,baseDim))*baseVal
        # place cluster centers
        for i in range(centers):
            # randomly sampling center points
            x1 = np.random.randint(lowx, highx)
            y1 = np.random.randint(lowy, highy)

            # set valid x and y coodrinates of cluster centers and boarder outlines
            xout0 = x1-outerCircle-innerWidth if x1-outerCircle-innerWidth >= int(0.05*baseDim) else int(0.05*baseDim)
            xout2 = x1+outerCircle+innerWidth if x1+outerCircle+innerWidth < int(0.95*baseDim-1) else int(0.95*baseDim-1)
            yout0 = y1-outerCircle-innerHeight if y1-outerCircle-innerHeight >= int(0.05*baseDim) else int(0.05*baseDim)
            yout2 = y1+outerCircle+innerHeight if y1+outerCircle+innerHeight < int(0.95*baseDim-1) else int(0.95*baseDim-1)
            
            xin0 = x1-innerWidth if x1-innerWidth >= int(0.15*baseDim) else int(0.15*baseDim)    # originally: 0 and baseDim-1 as limits
            xin2 = x1+innerWidth if x1+innerWidth < int(0.85*baseDim-1) else int(0.85*baseDim-1)
            yin0 = y1-innerHeight if y1-innerHeight >= int(0.15*baseDim) else int(0.15*baseDim)
            yin2 = y1+innerHeight if y1+innerHeight < int(0.85*baseDim-1) else int(0.85*baseDim-1)
            
            # set pixels within boarder region to one baseVal and pixels within cluster center to 2*baseVal
            plane[yout0:yout2, xout0:xout2] = plane[yout0:yout2, xout0:xout2] + baseVal
            plane[yin0:yin2, xin0:xin2] = plane[yin0:yin2, xin0:xin2] + 2*baseVal
            # fix new range of valid cluster center position
            if i == 0:
                lowx = x1-size*4 if x1-size*4 >= int(0.1*baseDim) else int(0.1*baseDim)
                highx = x1+size*4 if x1+size*4 < int(0.9*baseDim-1) else int(0.9*baseDim-1)
                lowy = y1-size*0.1 if y1-size*0.1 >= int(0.1*baseDim) else int(0.1*baseDim)
                highy = y1+size*0.1 if y1+size*0.1 < int(0.9*baseDim-1) else int(0.9*baseDim-1)
                
    else:
        #============================================
        # Default shape is "Spattered"
        #============================================
        # initialize parameters
        size, innerWidth, innerHeight, outerCircle, gaussianKernel, gaussianSigma, lowx, highx, lowy, highy = initializeParameters(\
            baseDim, sizeFactor, innerWidthFactor, innerHeightFactor, outerCircleFactor, gaussianKernelFactor, gaussianSigmaFactor)
        # place cluster centers
        plane = clusterPlacement(centers, lowx, highx, lowy, highy, innerWidth, innerHeight, outerCircle, baseDim, size, baseVal)
    
    # smoothing the plane
    thresholdedPlane = smoothingPlane(baseDim, plane, gaussianKernel, gaussianSigma, threshold)
    
    return thresholdedPlane
    
