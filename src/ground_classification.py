# coding: utf-8
from pathlib import Path
import laspy
import CSF
import numpy as np
from typing import List, Tuple
import argparse
import os
import shutil
from glob import glob


def read_las_files(input_path: Path) -> List[Tuple[Path, laspy.LasData]]:
    """Return list of (file_path, las_data) for a file or all LAS/LAZ files in a directory."""
    input_path = Path(input_path)
    las_files = []

    if input_path.is_file():
        las_files.append((input_path, laspy.read(input_path)))
    elif input_path.is_dir():
        for file in sorted(input_path.glob("*.las")) + sorted(input_path.glob("*.laz")):
            las_files.append((file, laspy.read(file)))
    else:
        raise ValueError(f"{input_path} is not a valid file or directory")

    return las_files


def set_extra_dimension(input_path:str, dimension_name:str, label:int):
    las = laspy.read(input_path)

    if dimension_name not in las.point_format.dimension_names:
        extra_dim = laspy.ExtraBytesParams(name=dimension_name, type='int32')
        las.add_extra_dim(extra_dim)

    las[dimension_name] = np.full(len(las.points), label, dtype=np.uint32)

    las.write(input_path)


def run_csf_for_file(
    input_file: Path,
    ground_dir: Path,
    non_ground_dir: Path,
    cloth_resolution: float = 0.5,
    rigidness: int = 2,
    time_step: float = 0.65,
    class_threshold: float = 0.5,
    iterations: int = 500,
    slope_smooth: bool = False
) -> Tuple[Path, Path]:
    """Run CSF ground filtering on a single LAS/LAZ file and return output paths."""
    las = laspy.read(input_file)

    xyz = np.vstack((las.x, las.y, las.z)).transpose()

    csf = CSF.CSF()
    csf.params.bSloopSmooth = slope_smooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.interations = iterations

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    ground_idx = np.asarray(ground, dtype=np.int64)
    non_ground_idx = np.asarray(non_ground, dtype=np.int64)

    original_pc_header = las.header.copy()
    header = laspy.LasHeader(point_format=original_pc_header.point_format, version=original_pc_header.version)

    # Ground points
    ground_las = laspy.LasData(header)
    ground_las.points = las.points[ground_idx]

    ground_output = ground_dir / f"{Path(input_file).stem}_ground.laz"
    ground_output.parent.mkdir(parents=True, exist_ok=True)
    ground_las.write(ground_output)

    set_extra_dimension(ground_output, 'PredSemantic', 0)

    # Non-ground points
    non_ground_las = laspy.LasData(header)
    non_ground_las.points = las.points[non_ground_idx]

    non_ground_output = non_ground_dir / f"{Path(input_file).stem}_non_ground.laz"
    non_ground_output.parent.mkdir(parents=True, exist_ok=True)
    non_ground_las.write(non_ground_output)

    return ground_output, non_ground_output


def run_csf(
    input_dir: str,
    output_dir: str,
    cloth_resolution: float = 0.05,
    rigidness: int = 2,
    time_step: float = 0.65,
    class_threshold: float = 0.02,
    iterations: int = 500,
    slope_smooth: bool = False
) -> Tuple[Path, Path]:
    """
    Run CSF on all LAS/LAZ files in a directory.
    Returns a tuple: (ground_output_path, non_ground_output_path)
    """
    output_dir = Path(output_dir)
    # Clean output_directory, if necessary
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Remove directory with all files and subdirectories
        os.mkdir(output_dir) # Create empty directory

    ground_dir = output_dir / "ground"
    ground_dir.mkdir(parents=True, exist_ok=True)
    
    non_ground_dir = output_dir / "non_ground"
    non_ground_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all LAZ files in input directory
    laz_files = glob(f'{input_dir}/*.laz')

    for f in laz_files:
        ground, non_ground = run_csf_for_file(
                                input_file=f,
                                ground_dir=ground_dir,
                                non_ground_dir=non_ground_dir,
                                cloth_resolution=cloth_resolution,
                                rigidness=rigidness,
                                time_step=time_step,
                                class_threshold=class_threshold,
                                iterations=iterations,
                                slope_smooth=slope_smooth
                            )
    
    return ground_dir, non_ground_dir


def main():
    parser = argparse.ArgumentParser(description="CSF ground filtering for LAS/LAZ files")

    parser.add_argument("--cloth-resolution", type=float, default=0.05,
                        help="Cloth resolution refers to the grid size of cloth which is use to cover the terrain. The bigger cloth resolution you have set, the coarser DTM you will get.")
    parser.add_argument("--rigidness", type=int, choices=[1, 2, 3],    
                        default=2, help=(
                            "Cloth rigidness preset controlling terrain type: "
                            "1 = steep/rugged (soft cloth), 2 = relief (medium), "
                            "3 = flat (rigid). Default: 2 (relief)."))
    parser.add_argument("--time-step", type=float, default=0.65,
                        help="Simulation time step controlling how far the cloth moves per iteration. Larger values speed up convergence but may reduce stability; smaller values improve accuracy but require more iterations.")
    parser.add_argument("--class-threshold", type=float, default=0.5,
                        help="Classification threshold refers to a threshold to classify the original point cloud into ground and non-ground parts based on the distances between original point cloud and the simulated terrain. 0.5 is adapted to most of scenes.")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Maximum iteration times of terrain simulation. 500 is enough for most of scenes.")
    parser.add_argument("--slope-smooth", action="store_true",
                        help="Enable slope smoothing to improve ground detection in steep or rugged terrain. Helps reduce misclassification on sharp elevation changes.")

    args = parser.parse_args()

    ground, non_ground =  run_csf(
                            input_path=args.input,
                            output_dir=args.output_dir,
                            cloth_resolution=args.cloth_resolution,
                            rigidness=args.rigidness,
                            time_step=args.time_step,
                            class_threshold=args.class_threshold,
                            iterations=args.iterations,
                            slope_smooth=args.slope_smooth
                        )

if __name__ == "__main__":
    main()

