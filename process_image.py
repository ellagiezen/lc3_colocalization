#!/usr/bin/env python
"""
Image processing pipeline for bacterial infection responses in cells.
Modularized functions from Jupyter Notebook cells 4-9 (Z-stack maximum intensity projection to Create Radii).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure, filters
from skimage.morphology import dilation


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
            expanded_mask = dilation(previous_mask)
            
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


def _save_visualization_images(df, mip_image, channel_dapi, channel_lc3, 
                             channel_brightfield, channel_mav, 
                             blob_contours, filename, n_steps):
    """
    Save visualization images for validation.
    
    Args:
        df: DataFrame containing blob data
        mip_image: Maximum intensity projection image
        channel_dapi: DAPI channel
        channel_lc3: LC3 channel
        channel_brightfield: Brightfield channel
        channel_mav: MAV channel
        blob_contours: List of blob contours
        filename: Original filename
        n_steps: Number of profile steps
    """
    # Create output directories
    out_dir = Path("./Out")
    overview_dir = out_dir / "Overview"
    
    # Convert filename to Path object for stem handling
    filename_path = Path(filename) if isinstance(filename, str) else filename
    clusters_dir = out_dir / "Clusters" / filename_path.stem
    
    overview_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove file extension for cleaner filenames
    base_filename = filename_path.stem
    
    # 1. Save overview image (2x2 subplots of all channels)
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    ax = axs.ravel()
    
    # DAPI
    ax[0].imshow(channel_dapi, cmap="gray")
    ax[0].set_title("DAPI (Max Intensity)")
    ax[0].axis("off")
    
    # MAV
    ax[1].imshow(channel_mav, cmap="gray")
    ax[1].set_title("MAV (Max Intensity)")
    ax[1].axis("off")
    
    # Brightfield
    ax[2].imshow(channel_brightfield, cmap="gray")
    ax[2].set_title("Brightfield (Max Intensity)")
    ax[2].axis("off")
    
    # LC3
    ax[3].imshow(channel_lc3, cmap="gray")
    ax[3].set_title("LC3 (Max Intensity)")
    ax[3].axis("off")
    
    plt.suptitle(f"Overview: {filename_path.name}", y=0.98, fontsize=16)
    plt.tight_layout()
    
    overview_path = overview_dir / f"{base_filename}.jpg"
    plt.savefig(overview_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved overview image to: {overview_path}")
    
    # 2. Save individual blob images
    for blob_idx, contour in enumerate(blob_contours):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left: MAV channel with blob contour
        ax[0].imshow(channel_mav, cmap="gray")
        ax[0].plot(contour[:, 1], contour[:, 0], color="lime", linewidth=2)
        ax[0].set_title(f"MAV Channel - Blob {blob_idx}")
        ax[0].axis("off")
        
        # Right: LC3 channel with blob contour and profile
        ax[1].imshow(channel_lc3, cmap="gray")
        ax[1].plot(contour[:, 1], contour[:, 0], color="lime", linewidth=2)
        
        # Add profile plot as inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax[1], width="40%", height="30%", loc='lower right')
        
        # Get profile for this blob
        profile_cols = [col for col in df.columns if col.startswith('profile_')]
        blob_profile = df.iloc[blob_idx][profile_cols].values
        
        steps = np.arange(len(blob_profile))
        inset_ax.plot(steps, blob_profile, color='purple', lw=2, marker='o')
        inset_ax.set_title('LC3 Profile', fontsize=8)
        inset_ax.set_xlabel('Step', fontsize=6)
        inset_ax.set_ylabel('Intensity', fontsize=6)
        inset_ax.grid(True, alpha=0.5)
        
        ax[1].set_title(f"LC3 Channel - Blob {blob_idx}")
        ax[1].axis("off")
        
        plt.suptitle(f"{filename_path.name} - Blob {blob_idx}", y=0.95, fontsize=14)
        # Note: tight_layout() removed due to incompatibility with inset axes
        
        blob_path = clusters_dir / f"blob_{blob_idx:03d}.jpg"
        plt.savefig(blob_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    print(f"Saved {len(blob_contours)} blob images to: {clusters_dir}")


def process_image_pipeline(image_data, filename=None, min_blob_size=10, max_blob_size=1000, n_steps=10, save_images=False):
    """
    Complete image processing pipeline from MIP to radius profiles.
    
    Args:
        image_data: 4D numpy array with shape (Z, Y, X, C) or (Z, C, Y, X)
        filename: Optional filename for the image (used in DataFrame and image saving)
        min_blob_size: Minimum blob area in pixels
        max_blob_size: Maximum blob area in pixels
        n_steps: Number of expansion steps for profiles
        save_images: If True, saves overview and individual blob images to ./Out/ folder
        
    Returns:
        pandas.DataFrame: DataFrame with one row per MAV blob containing:
            - filename: Image filename
            - blob_index: Blob index
            - blob_area: Blob area in pixels
            - x_position: X coordinate of centroid
            - y_position: Y coordinate of centroid
            - profile_0 to profile_{n_steps-1}: Intensity profile values
        
    Side effects:
        If save_images=True, creates:
        - ./Out/Overview/[filename].jpg - Overview of all channels
        - ./Out/Clusters/[filename]/blob_[index].jpg - Individual blob images
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
    
    # Create DataFrame with blob information
    blob_data = []
    
    for blob_idx in range(num_valid_blobs):
        # Get blob properties
        blob_label = blob_idx + 1  # Labels start from 1
        blob_mask = (clean_labeled_mav == blob_label)
        blob_area = np.sum(blob_mask)
        
        # Get centroid (convert from (y, x) to (x, y) for consistency)
        x_pos = centroids_x[blob_idx] if blob_idx < len(centroids_x) else np.nan
        y_pos = centroids_y[blob_idx] if blob_idx < len(centroids_y) else np.nan
        
        # Create row data
        row_data = {
            'filename': filename,
            'blob_index': blob_idx,
            'blob_area': blob_area,
            'x_position': x_pos,
            'y_position': y_pos
        }
        
        # Add profile columns
        for step in range(n_steps):
            row_data[f'profile_{step}'] = profile_matrix[blob_idx, step]
        
        blob_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(blob_data)
    
    # Save images if requested
    if save_images and filename:
        _save_visualization_images(df, mip_image, channel_dapi, channel_lc3, 
                                  channel_brightfield, channel_mav, 
                                  blob_contours, filename, n_steps)
    
    return df

