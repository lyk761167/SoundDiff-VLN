# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import numpy as np
import torch
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb


def load_metadata(parent_folder):
    """
    Load metadata (points and graph) from the specified parent folder.
    
    Args:
        parent_folder (str): Path to the folder containing points.txt and graph.pkl
        
    Returns:
        tuple: (points, graph) where points is a list of 3D coordinates and graph is the navigation graph
        
    Raises:
        FileExistsError: If graph.pkl file does not exist
    """
    # Define paths to points.txt and graph.pkl files
    points_file = os.path.join(parent_folder, 'points.txt')
    
    if "replica" in parent_folder:
        # For Replica dataset: load and process points with Replica-specific transformations
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        # Transform coordinate system: (x, y-1.5528907, -z)
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            -points_data[:, 2])
        )
    else:
        # For other datasets (e.g., Matterport3D): load and process points with different transformation
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        # Transform coordinate system: (x, y-1.5, -z)
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            -points_data[:, 2])
        )
    
    # Check if graph file exists
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        # Load the navigation graph from pickle file
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def _to_tensor(v):
    """
    Convert input to PyTorch tensor if it's not already a tensor.
    
    Args:
        v: Input data (torch.Tensor, numpy.ndarray, or other array-like)
        
    Returns:
        torch.Tensor: The converted tensor
    """
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def convert_semantic_object_to_rgb(x):
    """
    Convert semantic segmentation labels to RGB image using 40-color palette.
    
    Args:
        x (numpy.ndarray): 2D array of semantic labels (integers)
        
    Returns:
        numpy.ndarray: RGB image with shape (height, width, 3)
    """
    # Create a PIL image with mode "P" (palette mode)
    semantic_img = Image.new("P", (x.shape[1], x.shape[0]))
    
    # Apply the 40-color palette (d3_40_colors_rgb) to the image
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    
    # Put data into the image, using modulo 40 to ensure values are in [0, 39]
    semantic_img.putdata((x.flatten() % 40).astype(np.uint8))
    
    # Convert to RGB mode and then to numpy array
    semantic_img = np.array(semantic_img.convert("RGB"))
    return semantic_img