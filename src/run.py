#!/usr/bin/env python3
"""Main orchestrator script for the green volume estimation pipeline."""

import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
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

sys.path.append(os.path.abspath("py-rct/src"))

from py_rct.rayextract import run_batch_segmentation, run_raycloudtools_segmentation_steps

from ground_classification import run_csf
from calculate_green_volume import voxel_based_green_volume


@dataclass
class OrchestratorParams:
    input: Path = None
    output: Path = None
    task: str = "all"
    tiling: bool = False
    clear_output: bool = False

    # Tiling
    smart_tile_output_dir: Path = None
    smart_tile_length: int = 30
    smart_tile_buffer: int = 10
    smart_tile_skip_dimension_reduction: bool = False
    smart_tile_workers: int = 4
    smart_tile_resolution_1: float = 0.01
    smart_tile_resolution_2: float = 0.1

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
    csf_cloth_resolution: float = 0.05
    csf_rigidness: int = 2
    csf_time_step: float = 0.65
    csf_class_threshold: float = 0.5
    csf_iterations: int = 500
    csf_slope_smooth: bool = False

    # AOI area estimation
    area_shapefile: Path = None

    # Voxel-based green volume calculation
    gv_output_dir: Path = None
    gv_voxel_sizes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    gv_dimension: str = 'PredSemantic',
    gv_correction_file: Path = None


def make_json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    return obj


def write_metadata(params: OrchestratorParams) -> Path:
    params.output.mkdir(parents=True, exist_ok=True)
    metadata_path = params.output / "run_metadata.json"

    data = asdict(params)
    data = make_json_safe(data)

    data["run_started"] = datetime.now().isoformat()
    data["argv"] = sys.argv

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return metadata_path


def create_fully_segmented_point_cloud(ground_dir:str, non_ground_dir:str, merged_dir:str) -> str:
    ground_files = glob(os.path.join(ground_dir, "*_ground.laz"))

    merged_files = []

    for gf in ground_files:
        base_name = os.path.basename(gf).replace("_ground.laz", "")
        ngf = os.path.join(non_ground_dir, f"{base_name}_non_ground.laz")

        if not os.path.exists(ngf):
            print(f"Skipping {base_name}, no matching non-ground file.")
            continue

        clean_base = base_name.replace("_raycloud", "")
        merged_file = os.path.join(merged_dir, f"{clean_base}.laz")

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


def parse_cli_args() -> OrchestratorParams:
    """
    Parse CLI arguments and return an OrchestratorParams instance.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", 
        required=True, 
        type=Path,
        help="Input directory containing raw LAZ/LAS files."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        type=Path,
        help="Output directory."
    )
    parser.add_argument(
        "--task",
        default="all",
        help=("Task to perform: 'segmentation' (wood-leaf, ground and instance segmentation), 'voxelization' (voxel-based green volume calculation). Default is 'all'."),
    )
    parser.add_argument(
        "--tiling",
        action="store_true",
        help=("Activates the 3Dtrees Smart Tile pipeline (with cross-tile merging of tree instance IDs and semantic segmentation results). Use for re-tiling of existing tiles or for processing of very large point clouds."),
    )
    parser.add_argument("--tile-length", type=int, default=30,
                        help="Tiling parameter: Tile size in meters (default: 30).")
    parser.add_argument("--tile-buffer", type=int, default=10,
                        help="Tiling parameter: Buffer overlap in meters (default: 10).")
    parser.add_argument("--skip-dimension-reduction", action="store_true",
                        help="Smart tile parameter: Set to False (default) to reduce to X, Y, Z only for ~37 percent file size reduction (useful for raw pre-segmentation data), set to True for keeping all dimensions (for post-segmentation data).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Tiling parameter: Number of parallel workers for processing (default: 4).")
    parser.add_argument("--resolution-1", type=float, default=0.01,
        help="First subsampling resolution in meters for the tile task.")
    parser.add_argument("--resolution-2", type=float, default=0.1,
        help="First subsampling resolution in meters for the tile task.")
    
    # py-rct arguments
    parser.add_argument("--gradient", type=float, default=1.0,
                        help="RCT parameter: Gradient threshold for terrain extraction (default: 1.0).")
    parser.add_argument("--max-diameter", type=float, default=0.9,
                        help="RCT parameter: Maximum diameter (m) for tree instance segmentation (default: 0.9).")
    parser.add_argument("--crop-length", type=float, default=1.0,
                        help="RCT parameter: Distance from branch tip to reconstruct QSM (default: 1.0).")
    parser.add_argument("--distance-limit", type=float, default=1.0,
                        help="RCT parameter: Maximum distance between neighbor points in a tree (default: 1.0).")
    parser.add_argument("--height-min", type=float, default=2.0,
                        help="RCT parameter: Minimum height counted as tree (default: 2.0).")
    parser.add_argument("--girth-height-ratio", type=float, default=0.12,
                        help="RCT parameter: Proportion of tree height to estimate trunk girth (default: 0.12).")
    parser.add_argument("--global-taper", type=float, default=0.024,
                        help="RCT parameter: Global taper value (diameter per length) (default: 0.024).")
    parser.add_argument("--global-taper-factor", type=float, default=0.3,
                        help="RCT parameter: Factor for global taper (0-1) (default: 0.3).")
    parser.add_argument("--gravity-factor", type=float, default=0.3,
                        help="RCT parameter: Preference for vertical trees (default: 0.3).")
    parser.add_argument("-sd", "--split-distance", required=False, type=float, default=0.02,
        help="RCT parameter: Smaller values produce more, finer splits; larger values produce fewer, coarser splits")
    parser.add_argument("--branch-segmentation", action="store_true",
                        help="RCT parameter: Segment per branch if set; otherwise per tree.")
    # parser.add_argument("--grid-width", type=float, default=None,
    #                     help="RCT parameter: Assumed grid width of point cloud for cropping (default: none).")

    # CSF arguments
    parser.add_argument("--cloth-resolution", type=float, default=0.05,
                        help="CSF parameter: Cloth resolution refers to the grid size of cloth which is use to cover the terrain. The bigger cloth resolution you have set, the coarser DTM you will get.")
    parser.add_argument("--rigidness", type=int, choices=[1, 2, 3],    
                        default=2, help=(
                            "CSF parameter: Cloth rigidness preset controlling terrain type: "
                            "1 = steep/rugged (soft cloth), 2 = relief (medium), "
                            "3 = flat (rigid). Default: 2 (relief)."))
    parser.add_argument("--time-step", type=float, default=0.65,
                        help="CSF parameter: Simulation time step controlling how far the cloth moves per iteration. Larger values speed up convergence but may reduce stability; smaller values improve accuracy but require more iterations.")
    parser.add_argument("--class-threshold", type=float, default=0.5,
                        help="CSF parameter: Classification threshold refers to a threshold to classify the original point cloud into ground and non-ground parts based on the distances between original point cloud and the simulated terrain. 0.5 is adapted to most of scenes.")
    parser.add_argument("--iterations", type=int, default=500,
                        help="CSF parameter: Maximum iteration times of terrain simulation. 500 is enough for most of scenes.")
    parser.add_argument("--slope-smooth", action="store_true",
                        help="CSF parameter: Enable slope smoothing to improve ground detection in steep or rugged terrain. Helps reduce misclassification on sharp elevation changes.")

    # area calculation arguments
    parser.add_argument("-s","--shapefile", type=str,
                    help="Shapefile used for cropping the point cloud to AOI.")
    
    # voxelization arguments
    parser.add_argument("-v","--voxel-sizes", nargs="+", type=float, default=[0.1,0.2,0.3],
                    help="List of voxel sizes used for voxelization of the point cloud.")
    parser.add_argument("-d", "--dimension", default="PredSemantic", type=str,
        help="Dimension name for semantic class labels, e.g. 'PredSemantic' or 'Classification'.")
    parser.add_argument(
        "--correction-file", type=str,
        help="CSV file with correction factors for GV estimation: layer,voxel_size,factor"
    )
    
    parser.add_argument(
        "--clear-output",
        action="store_true", 
        help=("If True, intermediate files are deleted after processing, "
              "leaving only the final full segmentation results. "
              "Enable this to save disk space.")
    )

    args = parser.parse_args()

    return OrchestratorParams(
        input=args.input,
        output=args.output,
        task=args.task,
        tiling=args.tiling,
        clear_output=args.clear_output,

        smart_tile_output_dir=args.output / "retile",
        smart_tile_length=args.tile_length,
        smart_tile_buffer=args.tile_buffer,
        smart_tile_resolution_1=args.resolution_1,
        smart_tile_resolution_2=args.resolution_2,
        smart_tile_skip_dimension_reduction=args.skip_dimension_reduction,
        smart_tile_workers=args.workers,   

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
        csf_cloth_resolution=args.cloth_resolution,
        csf_rigidness=args.rigidness,
        csf_time_step=args.time_step,
        csf_class_threshold=args.class_threshold,
        csf_iterations=args.iterations,
        csf_slope_smooth=args.slope_smooth,

        area_shapefile=args.shapefile,

        gv_output_dir=args.output / "results",
        gv_voxel_sizes=args.voxel_sizes,
        gv_dimension=args.dimension,
        gv_correction_file=args.correction_file
    )

def main(params: OrchestratorParams):
    # Check given parameters
    if not params.task in ["segmentation", "voxelization", "all"]:
        print(f"Error: Unknown task: {params.task}")
        print("Valid tasks: all, segmentation, voxelization")
        sys.exit(1)

    if not Path(params.input).is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {params.input}")

    if not Path(params.area_shapefile).is_file():
        raise FileNotFoundError(f"Shapefile file does not exist: {params.area_shapefile}")
    
    # Ensure output dir exists
    os.makedirs(params.output, exist_ok=True)

    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()

    print("=" * 60)
    print("Running TLS Green Volume Pipeline")
    print("=" * 60)
    print(f"Input directory: {params.input}")
    print(f"Output directory: {params.output}")
    print(f"Task: {params.task}")
    print(f"Tiling: {params.tiling}")
    print(f"Processing started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not params.task == "voxelization":
        # Write metadata
        metadata_path = write_metadata(params)
        print(f"Run metadata written to {metadata_path}")

        # ----------------------------------------------
        # Step 1: Tiling
        # ----------------------------------------------
        print("\n" + "=" * 60)
        print("Tiling")
        print("=" * 60)

        if params.tiling:
            cmd = [
                "python", "3dtrees_Smart_Tile/src/run.py",
                "--task", "tile",
                "--input-dir", str(params.input),
                "--output-dir", str(params.smart_tile_output_dir),
                "--workers", str(params.smart_tile_workers),
                "--tile-length", str(params.smart_tile_length),
                "--tile-buffer", str(params.smart_tile_buffer),
                "--resolution-1", str(params.smart_tile_resolution_1),
                "--resolution-2", str(params.smart_tile_resolution_2),
                "--output-copc-res1", "False",
                "--dimension-reduction", "False"
            ]

            result = subprocess.run(cmd, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Tile task failed with code {result.returncode}")

            pyrct_input_path = params.smart_tile_output_dir / "subsampled_res1"
        else: 
            print("skipped.")
            pyrct_input_path = params.input


        # -------------------------------------------
        # Step 2: Semantic Segmentation
        # -------------------------------------------

        # RayCloudTools for leaf-wood segmentation
        print("\n" + "=" * 60)
        print("RayCloudTools")
        print("=" * 60)

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

        # CSF for ground classification
        print("\n" + "=" * 60)
        print("Cloth Simulation Filter")
        print("=" * 60)

        ground, non_ground = run_csf(
            input_dir=params.csf_input_path, 
            output_dir=params.csf_output_dir,
            cloth_resolution=params.csf_cloth_resolution,
            rigidness=params.csf_rigidness,
            time_step=params.csf_time_step,
            class_threshold=params.csf_class_threshold,
            iterations=params.csf_iterations,
            slope_smooth=params.csf_slope_smooth
        )

        # Create fully segmented point clouds
        print("\n" + "=" * 60)
        print("Merge segmentation results")
        print("=" * 60)

        merged_segmentation_dir = os.path.join(params.output, "results/segmented_laz")
        os.makedirs(merged_segmentation_dir, exist_ok=True)

        segmented_point_clouds = create_fully_segmented_point_cloud(
                ground_dir=ground, 
                non_ground_dir=non_ground,
                merged_dir=merged_segmentation_dir
            )
        
        print(f"Semantic segmentation results saved to\n " + "\n ".join(segmented_point_clouds))

        # Cross-tile merging
        if params.tiling:
            print("\n" + "=" * 60)
            print("Merge tiles")
            print("=" * 60)

            """
            source_dir = Path(params.pyrct_output_dir) / "trees"
            target_dir = Path(merged_segmentation_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            for txt_file in source_dir.glob("*_trees.txt"):
                shutil.copy2(txt_file, target_dir / txt_file.name)
            """

            cmd = [
                "python", "3dtrees_Smart_Tile/src/run.py",
                "--task", "filter",
                "--segmented-folders", str(merged_segmentation_dir),
                "--original-input-dir", str(params.input),
                "--remap-merge", "True",
                "--produce-merged-file", "False"
            ]

            result = subprocess.run(cmd, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Filter/Remap task failed with code {result.returncode}")

            # Clear output of tile merging step
            filtered_results_dir = params.output / "results" / "filtered"
            
            orig_predictions_dir = filtered_results_dir / "original_with_predictions"
            filtered_tiles_dir = filtered_results_dir / "filtered_tiles"
            
            if not orig_predictions_dir.is_dir():
                raise FileNotFoundError(f"Missing directory after tile merge: {orig_predictions_dir}")
            if not filtered_tiles_dir.is_dir():
                raise FileNotFoundError(f"Missing directory after tile merge: {filtered_tiles_dir}")
            
            if Path(merged_segmentation_dir).exists():
                shutil.rmtree(merged_segmentation_dir)

            shutil.move(str(orig_predictions_dir), str(merged_segmentation_dir))
            shutil.move(str(filtered_tiles_dir), str(Path(merged_segmentation_dir) / "segmented_buffered_tiles"))
            
            if Path(filtered_results_dir).exists():
                shutil.rmtree(filtered_results_dir)
            


    if not params.task == "segmentation":
        # -----------------------------------
        # Step 3: Calculate green volume
        # -----------------------------------
        print("\n" + "=" * 60)
        print("Voxel-based green volume calculation")
        print("=" * 60)
        print(f"Voxel sizes: {params.gv_voxel_sizes}")

        if params.task == "voxelization":
            gv_input_path = params.input
            dimension = params.gv_dimension
        elif params.task == "all":
            gv_input_path = merged_segmentation_dir
            dimension = 'PredSemantic'

        results = voxel_based_green_volume(
            input_path=gv_input_path,
            output_dir=params.gv_output_dir, 
            voxel_sizes=params.gv_voxel_sizes, 
            dimension=dimension,
            class_labels=[2,3], 
            shapefile=params.area_shapefile if params.area_shapefile else None,
            correction_file=params.gv_correction_file
        )

    if params.clear_output:
        print("\n" + "=" * 60)
        print("Clearing output directories")
        print("=" * 60)
        # Remove segmentation output
        for d in [params.pyrct_output_dir, params.csf_output_dir]:
            if d and d.is_dir():
                print(f"Removing {d}")
                shutil.rmtree(d)

    # End timing
    end_time = time.time()
    end_datetime = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print(f"Pipeline Complete ({end_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)
    print("\n" + f"Total processing time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")


if __name__ == "__main__":
    params = parse_cli_args()
    main(params)