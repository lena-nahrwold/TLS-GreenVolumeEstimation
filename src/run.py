#!/usr/bin/env python3
"""Main orchestrator script for the green volume estimation pipeline."""

import sys
import os
import argparse
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import List
from glob import glob
import subprocess
import json

import laspy
import numpy as np

from datetime import datetime


sys.path.append(os.path.abspath("../py-rct/src"))

from py_rct.rayextract import run_batch_segmentation, run_raycloudtools_segmentation_steps

from ground_classification import run_batch_csf, run_csf_for_file
from estimate_area_from_shp import estimate_area_from_shp
from calculate_green_volume import voxel_based_green_volume


@dataclass
class OrchestratorParams:
    output: Path = None

    # CSF
    csf_input_path: Path = None
    csf_output_dir: Path = None
    csf_ground_label: int = 0
    csf_non_ground_label: int = 1
    csf_cloth_resolution: float = 0.05
    csf_rigidness: int = 3
    csf_time_step: float = 0.65
    csf_class_threshold: float = 0.02
    csf_iterations: int = 500
    csf_slope_smooth: bool = False

    # py-rct
    pyrct_input_path: Path = None
    pyrct_output_dir: Path = None
    pyrct_gradient: float = 1.0
    pyrct_max_diameter: float = 0.9
    pyrct_crop_length: float = 1.0
    pyrct_distance_limit: float = 1.0
    pyrct_height_min: float = 2.0
    pyrct_girth_height_ratio: float = 0.12
    pyrct_global_taper: float = 0.024
    pyrct_global_taper_factor: float = 0.3
    pyrct_gravity_factor: float = 0.3
    pyrct_split_distance: float = 0.02,
    pyrct_branch_segmentation: bool = False
    # pyrct_grid_width: float = None

    # AOI area estimation
    area_shapefile: Path = None

    # Voxel-based green volume calculation
    gv_input_path: Path = None
    gv_output_dir: Path = None
    gv_voxel_sizes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


def parse_cli_args() -> OrchestratorParams:
    """
    Parse CLI arguments and return an OrchestratorParams instance.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=Path,
                    help="Input directory containing raw LAZ/LAS files.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory.")
    
    # py-rct arguments
    parser.add_argument("--gradient", type=float, default=1.0,
                        help="Gradient threshold for terrain extraction (default: 1.0).")
    parser.add_argument("--max-diameter", type=float, default=0.9,
                        help="Maximum diameter (m) for tree instance segmentation (default: 0.9).")
    parser.add_argument("--crop-length", type=float, default=1.0,
                        help="Distance from branch tip to reconstruct QSM (default: 1.0).")
    parser.add_argument("--distance-limit", type=float, default=1.0,
                        help="Maximum distance between neighbor points in a tree (default: 1.0).")
    parser.add_argument("--height-min", type=float, default=2.0,
                        help="Minimum height counted as tree (default: 2.0).")
    parser.add_argument("--girth-height-ratio", type=float, default=0.12,
                        help="Proportion of tree height to estimate trunk girth (default: 0.12).")
    parser.add_argument("--global-taper", type=float, default=0.024,
                        help="Global taper value (diameter per length) (default: 0.024).")
    parser.add_argument("--global-taper-factor", type=float, default=0.3,
                        help="Factor for global taper (0-1) (default: 0.3).")
    parser.add_argument("--gravity-factor", type=float, default=0.3,
                        help="Preference for vertical trees (default: 0.3).")
    parser.add_argument("-sd", "--split_distance", required=False, type=float, default=0.02,
        help="Smaller values produce more, finer splits; larger values produce fewer, coarser splits")
    parser.add_argument("--branch-segmentation", action="store_true",
                        help="Segment per branch if set; otherwise per tree.")
    # parser.add_argument("--grid-width", type=float, default=None,
    #                     help="Assumed grid width of point cloud for cropping (default: none).")

    # CSF
    # TODO

    # area calculation arguments
    #parser.add_argument("-s","--shapefile", required=True, type=str,
    #                help="Shapefile used for cropping the point cloud to AOI.")
    
    # voxelization arguments
    parser.add_argument("-v","--voxel-sizes", nargs="+", type=float, default=[0.1,0.2,0.3],
                    help="List of voxel sizes used for voxelization of the point cloud.")

    args = parser.parse_args()

    return OrchestratorParams(
        output=args.output,
        pyrct_input_path=args.input, 
        pyrct_output_dir=args.output / "rct_leaf_wood",
        pyrct_gradient=args.gradient,
        pyrct_max_diameter=args.max_diameter,
        pyrct_crop_length=args.crop_length,
        pyrct_distance_limit=args.distance_limit,
        pyrct_height_min=args.height_min,
        pyrct_girth_height_ratio=args.girth_height_ratio,
        pyrct_global_taper=args.global_taper,
        pyrct_global_taper_factor=args.global_taper_factor,
        pyrct_gravity_factor=args.gravity_factor,
        pyrct_split_distance=args.split_distance,
        pyrct_branch_segmentation=args.branch_segmentation,
        # pyrct_grid_width=args.grid_width,
        csf_input_path=args.output / "rct_leaf_wood" / "segmented" / "laz",
        csf_output_dir=args.output / "csf_ground",
        #area_shapefile=args.shapefile,
        gv_input_path=args.output / "csf_ground" / "non_ground",
        gv_output_dir=args.output / "results",
        gv_voxel_sizes=args.voxel_sizes
    )

def main(params: OrchestratorParams):
    # ----------------------------------------------
    # Step 1: Run py-rct for leaf-wood segmentation
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("RayCloudTools")
    print("=" * 60)

    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Processing started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    run_batch_segmentation(
        in_dir=params.pyrct_input_path,
        out_dir=params.pyrct_output_dir,
        gradient=params.pyrct_gradient,
        max_diameter=params.pyrct_max_diameter,
        crop_length=params.pyrct_crop_length,
        distance_limit=params.pyrct_distance_limit,
        height_min=params.pyrct_height_min,
        girth_height_ratio=params.pyrct_girth_height_ratio,
        global_taper=params.pyrct_global_taper,
        global_taper_factor=params.pyrct_global_taper_factor,
        gravity_factor=params.pyrct_gravity_factor,
        split_distance=params.pyrct_split_distance,
        branch_segmentation=params.pyrct_branch_segmentation,
        # pyrct_grid_width=args.grid_width
    )

    # End timing
    end_time = time.time()
    end_datetime = datetime.now()
    duration = end_time - start_time

    print(f"\nSegmentation steps completed at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

    # ------------------------------
    # Step 2: Ground classification
    # ------------------------------

    print("\n" + "=" * 60)
    print("Cloth Simulation Filter")
    print("=" * 60)

    if os.path.isdir(params.csf_input_path):
        ground, non_ground = run_batch_csf(
            input_path=params.csf_input_path, 
            output_dir=params.csf_output_dir
        )
    else:
        ground, non_ground = run_csf_for_file(
            input_file=params.csf_input_path,
            output_dir=params.csf_output_dir
        )

    # TODO
    ground_dir = os.path.join(params.csf_output_dir, "ground")
    non_ground_dir = os.path.join(params.csf_output_dir, "non_ground")
    merged_dir = os.path.join(params.output, "results")
    os.makedirs(merged_dir, exist_ok=True)

    ground_files = glob(os.path.join(ground_dir, "*_ground.laz"))

    for gf in ground_files:
        base_name = os.path.basename(gf).replace("_ground.laz", "")
        ngf = os.path.join(non_ground_dir, f"{base_name}_non_ground.laz")

        if not os.path.exists(ngf):
            print(f"Skipping {base_name}, no matching non-ground file.")
            continue

        merged_file = os.path.join(merged_dir, f"{base_name}_merged.laz")

        pipeline_dict = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": gf,
                    "tag": "ground"
                },
                {
                    "type": "readers.las",
                    "filename": ngf,
                    "tag": "nonground_raw"
                },
                {
                    "type": "filters.assign",
                    "assignment": "PredSemantic[0:0]=3",
                    "inputs": ["nonground_raw"],
                    "tag": "nonground"
                },
                {
                    "type": "filters.merge",
                    "inputs": ["ground", "nonground"],
                    "tag": "merged"
                },
                {
                    "type": "writers.las",
                    "filename": merged_file,
                    "inputs": ["merged"],
                    "minor_version": 4,
                    "extra_dims": "all"
                }
            ]
        }

        pipeline_json = json.dumps(pipeline_dict)

        subprocess.run(
            ["pdal", "pipeline", "--stdin"],
            input=pipeline_json.encode(),
            check=True
        )

        print(f"Merged file written: {merged_file}")

    # ------------------------------
    # Step 3: Estimate area size
    # ------------------------------

    print("\n" + "=" * 60)
    print("Area size")
    print("=" * 60)

    area = 0.02 #estimate_area_from_shp(shapefile=params.area_shapefile)

    # ------------------------------
    # Step 4: Calculate green volume
    # ------------------------------

    print("\n" + "=" * 60)
    print("Voxel-based green volume calculation")
    print("=" * 60)

    results = voxel_based_green_volume(
        input_path=params.gv_input_path,
        output_dir=params.gv_output_dir, 
        voxel_sizes=params.gv_voxel_sizes, 
        class_labels=[0,2], 
        area_size=area
    )

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print("\n" + f"Results saved to {results}.")



if __name__ == "__main__":
    params = parse_cli_args()
    main(params)