#!/usr/bin/env python3
"""Main orchestrator script for the green volume estimation pipeline."""

import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from glob import glob
import subprocess
import json
import shutil
import time
import laspy
from laspy.point.record import PackedPointRecord
import numpy as np

from datetime import datetime

os.environ['LC_ALL'] = 'C'

sys.path.append(os.path.abspath("/3dtrees_Smart_Tile/src"))
sys.path.append(os.path.abspath("/py-rct/src"))

from trees_smart_tile.run import run_tile_task, run_merge_task
from trees_smart_tile.parameters import Parameters
from py_rct.rayextract import run_batch_segmentation, run_raycloudtools_segmentation_steps

from ground_classification import run_csf
from calculate_green_volume import voxel_based_green_volume




@dataclass
class OrchestratorParams:
    input: Path = None
    output: Path = None
    basic_tiling: Path = False
    skip_tiling: bool = False
    clear_segmentation_output: bool = False

    # Tiling
    tile_output_dir: Path = None
    tile_original_dir: Path = None
    tile_length: int = 30
    tile_buffer: int = 10
    smart_tile_skip_dimension_reduction: bool = False

    # py-rct
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

    # Merging
    smart_merge_input_dir: Path = None
    smart_merge_tile_bounds_tindex: Path = None

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
    parser.add_argument(
        "--basic-tiling",
        action="store_true",
        help=(
            "Use fast, basic tiling without cross-tile merging of segmentation results."
            "For merged tree instance IDs, smart tiling is required."
        ),
    )
    parser.add_argument("--skip-tiling", action="store_true")
    parser.add_argument("--tile-length", type=int, default=30,
                        help="Tile size in meters for Smart Tile (default: 30).")
    parser.add_argument("--tile-buffer", type=int, default=10,
                        help="Buffer overlap in meters for Smart Tile tiles (default: 10).")
    parser.add_argument("--skip-dimension-reduction", type=bool, default=False,
                        help="Set to False (default) to reduce to X, Y, Z only for ~37 percent file size reduction (useful for raw pre-segmentation data), set to True for keeping all dimensions (for post-segmentation data).")
    
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
    parser.add_argument("-sd", "--split-distance", required=False, type=float, default=0.02,
        help="Smaller values produce more, finer splits; larger values produce fewer, coarser splits")
    parser.add_argument("--branch-segmentation", action="store_true",
                        help="Segment per branch if set; otherwise per tree.")
    # parser.add_argument("--grid-width", type=float, default=None,
    #                     help="Assumed grid width of point cloud for cropping (default: none).")

    # CSF
    # TODO

    # area calculation arguments
    parser.add_argument("-s","--shapefile", required=True, type=str,
                    help="Shapefile used for cropping the point cloud to AOI.")
    
    # voxelization arguments
    parser.add_argument("-v","--voxel-sizes", nargs="+", type=float, default=[0.1,0.2,0.3],
                    help="List of voxel sizes used for voxelization of the point cloud.")
    
    # TODO
    parser.add_argument(
        "--clear-segmentation-output",
        action="store_true",
        help=(
            "If True, intermediate segmentation files are deleted after processing, "
            "leaving only the final full segmentation results. "
            "Enable this to save disk space."
        )
    )

    args = parser.parse_args()

    return OrchestratorParams(
        input=args.input,
        output=args.output,
        basic_tiling=args.basic_tiling,
        skip_tiling=args.skip_tiling,
        clear_segmentation_output=args.clear_segmentation_output,

        tile_output_dir=args.output / "tiles",
        tile_original_dir=args.input,
        tile_length=args.tile_length,
        tile_buffer=args.tile_buffer,
        smart_tile_skip_dimension_reduction=args.skip_dimension_reduction,   

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

        smart_merge_input_dir= args.output / "results" / "segmented_laz",
        smart_merge_tile_bounds_tindex= args.output / "tiles" / "tile_bounds_tindex.json",

        area_shapefile=args.shapefile,

        gv_input_path=args.output / "csf_ground" / "non_ground",
        gv_output_dir=args.output / "results",
        gv_voxel_sizes=args.voxel_sizes
    )


def run_basic_tiling(input: str, output_dir: str, tile_length: int, buffer: int) -> list:
    # Check that output directory exists
    if not os.path.exists(output_dir):
        # If not, create it
        os.makedirs(output_dir)

    subprocess.run(
        [
            "pdal", "tile", input, str(output_dir /"tile_#.laz"), 
            "--length", str(tile_length), 
            "--buffer", str(buffer)
        ],
        check=True
    )

    return output_dir


def create_fully_segmented_point_cloud(ground_dir:str, non_ground_dir:str, merged_dir:str) -> str:
    ground_files = glob(os.path.join(ground_dir, "*_ground.laz"))

    merged_files = []

    for gf in ground_files:
        base_name = os.path.basename(gf).replace("_ground.laz", "")
        ngf = os.path.join(non_ground_dir, f"{base_name}_non_ground.laz")

        if not os.path.exists(ngf):
            print(f"Skipping {base_name}, no matching non-ground file.")
            continue

        merged_file = os.path.join(merged_dir, f"{base_name}_merged.laz")

        ground = laspy.read(gf)
        nonground = laspy.read(ngf)

        dimension_name = "PredSemantic"

        if dimension_name not in nonground.point_format.dimension_names:
            raise ValueError(f"{dimension_name} not found in non-ground file")

        # Replace 0 → 3 (low vegetation)
        mask = nonground[dimension_name] == 0
        nonground[dimension_name][mask] = 3

        if ground.point_format.id != nonground.point_format.id:
            raise ValueError("Point format mismatch")
        if tuple(ground.point_format.extra_dimension_names) != tuple(nonground.point_format.extra_dimension_names):
            raise ValueError("Extra dims mismatch")

        # Concatenate underlying numpy arrays
        merged_array = np.concatenate([ground.points.array, nonground.points.array])

        # Build a new PackedPointRecord with the same point_format
        merged_points = PackedPointRecord(merged_array, ground.point_format)

        # Create output LasData from ground header and assign points
        merged = laspy.LasData(ground.header.copy())
        merged.points = merged_points

        merged.write(merged_file)

        merged_files.append(merged_file)

    return merged_files

def main(params: OrchestratorParams):
    # Check that output directory exists
    if not os.path.exists(params.output):
        # If not, create it
        os.makedirs(params.output)

    # ----------------------------------------------
    # Step 1: Tiling
    # ----------------------------------------------
    """
    if not params.skip_tiling:
        if params.basic_tiling:
            pyrct_input_path = run_basic_tiling(input=params.input, 
                                            output_dir=params.tile_output_dir,
                                            tile_length=params.tile_length,
                                            buffer=params.tile_buffer)
        else:
            smart_tile_params = Parameters(
                input_dir=params.input,
                output_dir=params.tile_output_dir,
                tile_length=params.tile_length,
                tile_buffer=params.tile_buffer,
                skip_dimension_reduction=False,
                workers=8 # TODO
            )

            run_tile_task(smart_tile_params)

            pyrct_input_path = params.tile_output_dir / "subsampled_res1"
    
    else: 
        pyrct_input_path = params.input

    # ----------------------------------------------
    # Step 2: Run py-rct for leaf-wood segmentation
    # ----------------------------------------------
    print("\n" + "=" * 60)
    print("RayCloudTools")
    print("=" * 60)

    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Processing started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    run_batch_segmentation(
        in_dir=pyrct_input_path,
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
    # Step 3: Ground classification
    # ------------------------------

    print("\n" + "=" * 60)
    print("Cloth Simulation Filter")
    print("=" * 60)

    ground, non_ground = run_csf(
        input_dir=params.csf_input_path, 
        output_dir=params.csf_output_dir
    )

    # -------------------------------------------
    # Step 4: Create fully segmented point cloud
    # -------------------------------------------

    print("\n" + "=" * 60)
    print("Merge segmentation results")
    print("=" * 60)
    
    # TODO
    merged_dir = os.path.join(params.output, "results/segmented_laz")
    os.makedirs(merged_dir, exist_ok=True)

    segmented_point_clouds = create_fully_segmented_point_cloud(
            ground_dir=ground, 
            non_ground_dir=non_ground,
            merged_dir=merged_dir
        )

    # ----------------------------------------------
    # Step 5: Merging
    # ----------------------------------------------

    if not params.skip_tiling and not params.basic_tiling:
        smart_tile_params = Parameters(
            segmented_remapped_folder=params.smart_merge_input_dir,
            original_input_dir=params.smart_tile_original_dir,
            tile_bounds_json=params.smart_merge_tile_bounds_tindex,
            buffer=params.smart_tile_buffer,
            enable_volume_merge=False,
            skip_merged_file=True
        )

        run_merge_task(smart_tile_params)
    """
    # ------------------------------
    # Step 6: Calculate green volume
    # ------------------------------

    print("\n" + "=" * 60)
    print("Voxel-based green volume calculation")
    print("=" * 60)

    results = voxel_based_green_volume(
        input_path=params.gv_input_path,
        output_dir=params.gv_output_dir, 
        voxel_sizes=params.gv_voxel_sizes, 
        class_labels=[0,2], 
        shapefile=params.area_shapefile
    )

    if params.clear_segmentation_output:
        # Remove segmentation outputs
        dirs_to_remove = [params.pyrct_output_dir, params.csf_output_dir]
        for d in dirs_to_remove:
            shutil.rmtree(d)

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print("\n" + f"Results saved to {results}.")



if __name__ == "__main__":
    params = parse_cli_args()
    main(params)