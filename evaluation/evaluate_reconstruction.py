import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pathlib
import subprocess
import os
import shutil
import operator
import glob
import csv
import re
import json
import utils
from enum import Enum
from tqdm import tqdm
from skimage import io
from plyfile import PlyData, PlyElement
from DeepDeform_Eval import evaluate_deform as evaluate_deform_c, evaluate_geometry as evaluate_geometry_c
import config as cfg


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
DEBUG = True
USE_CACHE = True


class Status(Enum):
    SUCCESS = 0
    INCONSISTENT_MESHES = 1


def load_pointcloud(mesh_path, mesh_cache):
    if mesh_path in mesh_cache:
        plydata = mesh_cache[mesh_path]
    
    else:
        with open(mesh_path, 'rb') as f:
            plydata = PlyData.read(f)

        if USE_CACHE: mesh_cache[mesh_path] = plydata

    vertex_x = np.expand_dims(plydata['vertex']['x'], axis=1)
    vertex_y = np.expand_dims(plydata['vertex']['y'], axis=1)
    vertex_z = np.expand_dims(plydata['vertex']['z'], axis=1)
    vertex_coords = np.concatenate((vertex_x, vertex_y, vertex_z), axis=1) 

    return vertex_coords


def evaluate(
    groundtruth_matches_json_path, groundtruth_masks_json_path, 
    reconstructions_dir, sequence_data_dir, segment_length=100
):
    print("#"*60)
    print("RECONSTRUCTION EVALUATION")
    print("#"*60)

    print()
    print("Groundtruth matches: {}".format(groundtruth_matches_json_path))
    print("Reconstructions dir: {}".format(reconstructions_dir))
    print("Sequence data dir: {}".format(sequence_data_dir))
    print()

    mesh_cache = {}

    max_geometry_error = 0.30
    max_deform_error = 0.30

    print("-> Matches processing")

    # Load groundtruth matches.
    with open(groundtruth_matches_json_path, 'r') as f:
        matches = json.load(f)

    # Compute groundtruth matches per sequence.
    matches_per_seq = {}
    for frame_pair in matches:
        seq_id = frame_pair["seq_id"]
        frame_pair_matches = frame_pair["matches"]
       
        num_matches = len(frame_pair_matches)
        if num_matches == 0: continue

        source_pixels = np.zeros((num_matches, 2), dtype=np.float32)
        target_pixels = np.zeros((num_matches, 2), dtype=np.float32)
        for i, match in enumerate(frame_pair_matches):
            source_pixels[i, 0] = match["source_x"]
            source_pixels[i, 1] = match["source_y"]
            target_pixels[i, 0] = match["target_x"]
            target_pixels[i, 1] = match["target_y"]

        refined_frame_pair = {
            "seq_id": frame_pair["seq_id"],
            "object_id": frame_pair["object_id"],
            "source_id": frame_pair["source_id"],
            "target_id": frame_pair["target_id"],
            "source_color": frame_pair["source_color"],
            "source_depth": frame_pair["source_depth"],
            "target_color": frame_pair["target_color"],
            "target_depth": frame_pair["target_depth"],
            "source_pixels": source_pixels,
            "target_pixels": target_pixels
        }

        if seq_id in matches_per_seq:
            matches_per_seq[seq_id].append(refined_frame_pair)
        else:
            matches_per_seq[seq_id] = [refined_frame_pair]

    print("<- Matches processing")

    print("-> Masks processing")
    
    # Load groundtruth masks.
    with open(groundtruth_masks_json_path, 'r') as f:
        masks = json.load(f)

    # Compute groundtruth masks per sequence.
    masks_per_seq = {}
    for mask_details in masks:
        seq_id = mask_details["seq_id"]

        if seq_id in masks_per_seq:
            masks_per_seq[seq_id].append(mask_details)
        else:
            masks_per_seq[seq_id] = [mask_details]

    print("<- Masks processing")

    # Evaluate reconstruction on every sequence separately.
    seq_ids = sorted(matches_per_seq.keys())
    geometry_error_per_seq = {}
    deform_error_per_seq = {}

    print("-> Sequence processing")
    for seq_id in seq_ids:
        print("Processing sequence {}".format(seq_id))
        seq_dir = os.path.join(sequence_data_dir, seq_id)
        seq_frame_pairs = matches_per_seq[seq_id]
        seq_masks = masks_per_seq[seq_id]
        depth_dir = os.path.join(seq_dir, "depth")
        mask_dir = os.path.join(seq_dir, "mask")

        # Reset mesh cache, since the previous sequence's meshes are
        # not needed anymore.
        mesh_cache = {}

        # Load intrinsics.
        intrinsics_path = os.path.join(seq_dir, "intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_path)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        depth_normalizer = 1000.0

        # Evaluate current sequence.
        sequence_geometry_dist_sum = 0.0
        sequence_geometry_num_valid = 0.0
        sequence_deform_dist_sum = 0.0
        sequence_deform_num_valid = 0.0
      
        num_frames = len(os.listdir(depth_dir))
        end_idx = num_frames - 1

        # Execute evaluation segment-wise.        
        segment_end_idx = 0
        while segment_end_idx < end_idx:
            segment_end_idx = min(segment_end_idx + segment_length, end_idx) 
            segment_id = "{0}".format(segment_end_idx)

            ##########################################################
            # Geometry evaluation
            ##########################################################
            segment_geometry_dist_sum = 0.0
            segment_geometry_num_valid = 0.0

            for mask_details in seq_masks:
                frame_id = mask_details["frame_id"]

                frame_idx = int(frame_id)

                # We only evaluate masks that are inside the current segment.
                if frame_idx > segment_end_idx:
                    continue

                # Load depth map.
                depth_path = os.path.join(depth_dir, "{}.png".format(frame_id))
                assert os.path.exists(depth_path), "Depth doesn't exist!"

                depth = io.imread(depth_path)

                # Load segmentation mask.
                mask_path = os.path.join(mask_dir, "{}.png".format(frame_id))
                assert os.path.exists(mask_path), "Mask doesn't exist!"

                mask = io.imread(mask_path)

                # Load estimated mesh.
                mesh_path = os.path.join(reconstructions_dir, "{0}_{1}_{2}.ply".format(seq_id, segment_id, frame_id))
                if not os.path.exists(mesh_path):
                    # If mesh doesn't exist, we take the max possible error for the
                    # current frame pair.
                    print("Mesh doesn't exists!")
                    segment_geometry_dist_sum += max_geometry_error
                    segment_geometry_num_valid += 1
                    continue

                vertices = load_pointcloud(mesh_path, mesh_cache)
                num_vertices = vertices.shape[0]

                # Execute geometry evaluation.
                geometry_dist_sum_np = np.zeros((1), dtype=np.float32)
                geometry_num_valid_np = np.zeros((1), dtype=np.int32)
                evaluate_geometry_c(
                    depth, mask, vertices,
                    fx, fy, cx, cy, depth_normalizer,
                    max_geometry_error,
                    geometry_dist_sum_np, geometry_num_valid_np
                )

                geometry_dist_sum = float(geometry_dist_sum_np[0]) 
                geometry_num_valid = int(geometry_num_valid_np[0])

                # Update segment statistics.
                segment_geometry_dist_sum += geometry_dist_sum
                segment_geometry_num_valid += geometry_num_valid

            ##########################################################
            # Deformation evaluation
            ##########################################################
            segment_deform_dist_sum = 0.0
            segment_deform_num_valid = 0.0

            for frame_pair in seq_frame_pairs:
                source_id = frame_pair["source_id"]
                target_id = frame_pair["target_id"]

                source_idx = int(source_id)
                target_idx = int(target_id)

                # We only evaluate frame pairs that are inside the current segment.
                if source_idx > segment_end_idx or target_idx > segment_end_idx:
                    continue

                # Load depth maps.
                source_depth_path = os.path.join(depth_dir, "{}.png".format(source_id))
                target_depth_path = os.path.join(depth_dir, "{}.png".format(target_id))
                assert os.path.exists(source_depth_path), "Source depth doesn't exist!"
                assert os.path.exists(target_depth_path), "Target depth doesn't exist!"

                source_depth = io.imread(source_depth_path)
                target_depth = io.imread(target_depth_path)

                # Load segmentation masks.
                source_mask_path = os.path.join(mask_dir, "{}.png".format(source_id))
                target_mask_path = os.path.join(mask_dir, "{}.png".format(target_id))
                assert os.path.exists(source_mask_path), "Source mask doesn't exist!"
                assert os.path.exists(target_mask_path), "Target mask doesn't exist!"

                source_mask = io.imread(source_mask_path)
                target_mask = io.imread(target_mask_path)

                # Load estimated meshes.
                source_mesh_path = os.path.join(reconstructions_dir, "{0}_{1}_{2}.ply".format(seq_id, segment_id, source_id))
                target_mesh_path = os.path.join(reconstructions_dir, "{0}_{1}_{2}.ply".format(seq_id, segment_id, target_id))
                if not os.path.exists(source_mesh_path) or \
                    not os.path.exists(target_mesh_path):
                    # If meshes don't exist, we take the max possible error for the
                    # current frame pair.
                    print("Meshes don't exists!")
                    segment_deform_dist_sum += max_deform_error
                    segment_deform_num_valid += 1
                    continue

                source_vertices = load_pointcloud(source_mesh_path, mesh_cache)
                target_vertices = load_pointcloud(target_mesh_path, mesh_cache)

                num_vertices = source_vertices.shape[0]
                if target_vertices.shape[0] != num_vertices:
                    print("Inconsistent meshes!")
                    print("Source mesh: {}".format(source_mesh_path))                
                    print("Target mesh: {}".format(target_mesh_path))            
                    return { "status": Status.INCONSISTENT_MESHES }

                # Prepare matches.
                source_pixels = frame_pair["source_pixels"]
                target_pixels = frame_pair["target_pixels"]

                # Execute deformation evaluation.
                deform_dist_sum_np = np.zeros((1), dtype=np.float32)
                deform_num_valid_np = np.zeros((1), dtype=np.int32)
                evaluate_deform_c(
                    source_depth, target_depth,
                    source_mask, target_mask,
                    source_vertices, target_vertices,
                    source_pixels, target_pixels,
                    fx, fy, cx, cy, depth_normalizer,
                    max_deform_error,
                    deform_dist_sum_np, deform_num_valid_np
                )

                deform_dist_sum = float(deform_dist_sum_np[0])
                deform_num_valid = int(deform_num_valid_np[0])

                # Update segment statistics.
                segment_deform_dist_sum += deform_dist_sum
                segment_deform_num_valid += deform_num_valid

            # Compute average segment error and update sequence statistics.
            # There could be segments where we don't have any groundtruth matches.
            # We don't count these segments as valid, so they don't contribute to
            # the final sequence average.
            if segment_geometry_num_valid > 0:
                segment_geometry_mean = segment_geometry_dist_sum / segment_geometry_num_valid
                segment_geometry_mean = min(segment_geometry_mean, max_geometry_error)

                sequence_geometry_dist_sum += segment_geometry_mean
                sequence_geometry_num_valid += 1

            if segment_deform_num_valid > 0:
                segment_deform_mean = segment_deform_dist_sum / segment_deform_num_valid
                segment_deform_mean = min(segment_deform_mean, max_deform_error)

                sequence_deform_dist_sum += segment_deform_mean
                sequence_deform_num_valid += 1

            if DEBUG:
                if sequence_geometry_num_valid > 0:
                    print("\tSegment {0} geometry mean: {1}".format(segment_id, segment_geometry_mean))
                else:
                    print("\tSegment {0} geometry mean: {1}".format(segment_id, -1.0))

                if sequence_deform_num_valid > 0:
                    print("\tSegment {0} deform mean: {1}".format(segment_id, segment_deform_mean))
                else:
                    print("\tSegment {0} deform mean: {1}".format(segment_id, -1.0))
           
        # Store sequence statistics.
        assert sequence_geometry_num_valid > 0
        geometry_error_per_seq[seq_id] = sequence_geometry_dist_sum / sequence_geometry_num_valid

        assert sequence_deform_num_valid > 0
        deform_error_per_seq[seq_id] = sequence_deform_dist_sum / sequence_deform_num_valid

        if DEBUG:
            print("\tSequence {0} geometry mean: {1}".format(seq_id, geometry_error_per_seq[seq_id]))
            print("\tSequence {0} deform mean: {1}".format(seq_id, deform_error_per_seq[seq_id]))
        
    print("<- Sequence processing")

    # Compute total statistics.
    print()
    print("{0:<20s} | {1:^20s} | {2:^20s}".format("Sequence ID", "Deform. error (m)", "Geometry error (m)"))
    print("-"*66)

    geometry_error_sum = 0.0
    deform_error_sum = 0.0

    for seq_id in seq_ids:
        geometry_error_per_seq[seq_id] = 100.0 * geometry_error_per_seq[seq_id] # conversion to cm
        deform_error_per_seq[seq_id] = 100.0 * deform_error_per_seq[seq_id] # conversion to cm

        geometry_error_seq = geometry_error_per_seq[seq_id]
        deform_error_seq = deform_error_per_seq[seq_id]

        geometry_error_sum += geometry_error_seq
        deform_error_sum += deform_error_seq

        print("{0:<20s} | {1:^20.4f} | {2:^20.4f}".format(seq_id, deform_error_seq, geometry_error_seq))

    geometry_error = geometry_error_sum / len(seq_ids)
    deform_error = deform_error_sum / len(seq_ids)
    print("-"*66)
    print("{0:<20s} | {1:^20.4f} | {2:^20.4f}".format("Total", deform_error, geometry_error))

    return {
        "status": Status.SUCCESS, 
        "deform_error": deform_error, 
        "geometry_error": geometry_error, 
        "deform_error_per_seq": deform_error_per_seq, 
        "geometry_error_per_seq": geometry_error_per_seq
    }
    

if __name__ == "__main__":
    groundtruth_matches_json_path = os.path.join(cfg.DATA_ROOT_DIR, "{0}_matches.json".format(cfg.DATA_TYPE))
    groundtruth_masks_json_path = os.path.join(cfg.DATA_ROOT_DIR, "{0}_masks.json".format(cfg.DATA_TYPE))
    reconstructions_dir = cfg.RECONSTRUCTION_RESULTS_DIR
    sequence_data_dir = os.path.join(cfg.DATA_ROOT_DIR, cfg.DATA_TYPE)

    evaluate(
        groundtruth_matches_json_path, groundtruth_masks_json_path,
        reconstructions_dir, sequence_data_dir
    )