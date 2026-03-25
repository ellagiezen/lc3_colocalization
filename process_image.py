#!/usr/bin/env python
"""
Image processing pipeline for bacterial infection responses in cells.
Modularized functions from Jupyter Notebook cells 4-9 (Z-stack maximum intensity projection to Create Radii).
"""

import numpy as np
from skimage import measure, filters
from skimage.morphology import binary_dilation


def perform_mip(image_data):
    """
    Perform Maximum Intensity Projection (MIP) on Z-stack image data.
    
    Args:
        image_data: 4D numpy array with shape (Z, Y, X, C) or (Z, C, Y, X)
        
    Returns:
        tuple: (mip_image, channel_dapi, channel_lc3, channel_brightfield, channel_mav)
    """
    # Perform MIP along Z-axis (axis 0)
    mip_image = np.max(image_data, axis=0)
    
    # Extract channels - assuming last dimension is channels
    # Shape should be (Y, X, C) after MIP
    channel_dapi = mip_image[:, :, 0]
    channel_lc3 = mip_image[:, :, 1]
    channel_brightfield = mip_image[:, :, 2]
    channel_mav = mip_image[:, :, 3]
    
    return mip_image, channel_dapi, channel_lc3, channel_brightfield, channel_mav


def detect_mav_clusters(channel_mav, min_blob_size=10, max_blob_size=1000):
    """
    Detect MAV clusters using Otsu thresholding and size filtering.
    
    Args:
        channel_mav: 2D numpy array containing MAV channel data
        min_blob_size: Minimum blob area in pixels
        max_blob_size: Maximum blob area in pixels
        
    Returns:
        tuple: (binary_mav, labeled_mav, centroids_x, centroids_y, blob_contours)
    """
    # Create binary image using Otsu thresholding
    thresh_mav = filters.threshold_otsu(channel_mav)
    binary_mav = channel_mav > thresh_mav
    
    # Label connected components
    labeled_mav = measure.label(binary_mav)
    
    # Initialize lists to store blob data
    centroids_x, centroids_y = [], []
    blob_contours = []
    
    # Filter by size and trace boundaries
    for prop in measure.regionprops(labeled_mav):
        if min_blob_size <= prop.area <= max_blob_size:
            # Store centroid
            y, x = prop.centroid
            centroids_y.append(y)
            centroids_x.append(x)
            
            # Trace boundary contour
            object_mask = (labeled_mav == prop.label)
            contours = measure.find_contours(object_mask, 0.5)
            
            if contours:
                blob_contours.append(contours[0])
    
    print(f"Successfully identified and traced {len(blob_contours)} blobs.")
    
    return binary_mav, labeled_mav, centroids_x, centroids_y, blob_contours


def create_clean_labeled_image(labeled_mav, min_blob_size=10, max_blob_size=1000):
    """
    Create a clean labeled image with only valid blobs.
    
    Args:
        labeled_mav: Labeled image from measure.label
        min_blob_size: Minimum blob area in pixels
        max_blob_size: Maximum blob area in pixels
        
    Returns:
        tuple: (clean_labeled_mav, num_valid_blobs)
    """
    # Start with blank canvas
    clean_labeled_mav = np.zeros_like(labeled_mav)
    valid_label_counter = 1
    
    # Filter and relabel valid blobs
    for prop in measure.regionprops(labeled_mav):
        if min_blob_size <= prop.area <= max_blob_size:
            clean_labeled_mav[labeled_mav == prop.label] = valid_label_counter
            valid_label_counter += 1
    
    num_valid = valid_label_counter - 1
    print(f"Cleaned image contains {num_valid} blobs.")
    
    return clean_labeled_mav, num_valid


def create_radii_profiles(clean_labeled_mav, channel_lc3, n_steps=10):
    """
    Create area-normalized intensity profiles around MAV blobs.
    
    Args:
        clean_labeled_mav: Clean labeled image with valid blobs
        channel_lc3: LC3 channel data
        n_steps: Number of expansion steps (including interior)
        
    Returns:
        numpy array: Profile matrix with shape (num_blobs, n_steps)
    """
    num_blobs = int(clean_labeled_mav.max())
    profile_matrix = np.zeros((num_blobs, n_steps))
    
    print(f"Generating {n_steps}-step area-normalized profiles for {num_blobs} blobs...")
    
    for i in range(1, num_blobs + 1):
        # Isolate current blob
        current_mask = (clean_labeled_mav == i)
        blob_area = np.sum(current_mask)
        
        # Safety check for zero area
        if blob_area == 0:
            profile_matrix[i-1, :] = np.nan
            continue
        
        # Step 0: Interior of the trace
        internal_values = channel_lc3[current_mask]
        if len(internal_values) > 0:
            profile_matrix[i-1, 0] = np.mean(internal_values) / blob_area
        else:
            profile_matrix[i-1, 0] = np.nan
        
        # Steps 1 to n_steps-1: Expanding outwards
        previous_mask = current_mask
        for step in range(1, n_steps):
            # Grow the shape by 1 pixel
            expanded_mask = binary_dilation(previous_mask)
            
            # Identify the new ring
            ring_mask = expanded_mask ^ previous_mask
            
            # Sample LC3 intensity from this ring
            ring_values = channel_lc3[ring_mask]
            
            if len(ring_values) > 0:
                profile_matrix[i-1, step] = np.mean(ring_values) / blob_area
            else:
                profile_matrix[i-1, step] = np.nan
            
            previous_mask = expanded_mask
    
    print("Profile Matrix Generation Complete.")
    print(f"Shape: {profile_matrix.shape} (Row: Blob, Column: Distance from Trace)")
    
    return profile_matrix


def process_image_pipeline(image_data, min_blob_size=10, max_blob_size=1000, n_steps=10):
    """
    Complete image processing pipeline from MIP to radius profiles.
    
    Args:
        image_data: 4D numpy array with shape (Z, Y, X, C) or (Z, C, Y, X)
        min_blob_size: Minimum blob area in pixels
        max_blob_size: Maximum blob area in pixels
        n_steps: Number of expansion steps for profiles
        
    Returns:
        dict: Dictionary containing all processing results
    """
    # Step 1: Maximum Intensity Projection
    mip_image, channel_dapi, channel_lc3, channel_brightfield, channel_mav = perform_mip(image_data)
    
    # Step 2: MAV cluster detection
    binary_mav, labeled_mav, centroids_x, centroids_y, blob_contours = detect_mav_clusters(
        channel_mav, min_blob_size, max_blob_size
    )
    
    # Step 3: Create clean labeled image
    clean_labeled_mav, num_valid_blobs = create_clean_labeled_image(
        labeled_mav, min_blob_size, max_blob_size
    )
    
    # Step 4: Create radius profiles
    profile_matrix = create_radii_profiles(clean_labeled_mav, channel_lc3, n_steps)
    
    return {
        'mip_image': mip_image,
        'channels': {
            'dapi': channel_dapi,
            'lc3': channel_lc3,
            'brightfield': channel_brightfield,
            'mav': channel_mav
        },
        'binary_mav': binary_mav,
        'labeled_mav': labeled_mav,
        'clean_labeled_mav': clean_labeled_mav,
        'centroids': {'x': centroids_x, 'y': centroids_y},
        'blob_contours': blob_contours,
        'profile_matrix': profile_matrix,
        'num_valid_blobs': num_valid_blobs
    }

