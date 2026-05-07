import argparse
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d


def read_laz_raw(filename: Path):
    las = laspy.read(filename)
    pts = np.vstack((las.x, las.y, las.z)).T
    return las, pts


def to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def add_predsemantic_dimension(las: laspy.LasData):
    if "PredSemantic" in las.point_format.dimension_names:
        return 

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="PredSemantic",
            type=np.uint8,
            description="Predicted semantic label"
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute leaf/wood separation between leaf-on and leaf-off LAS/LAZ and optionally write PredSemantic labels."
    )
    parser.add_argument(
        "--leaf-on", "-l",
        required=True,
        type=Path,
        help="Leaf-on LAS/LAZ file"
    )
    parser.add_argument(
        "--leaf-off", "-k",
        required=True,
        type=Path,
        help="Leaf-off LAS/LAZ file"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.02,
        help="Distance threshold to distinguish leaf vs wood (default: 0.02)"
    )
    parser.add_argument(
        "--out-leaf",
        type=Path,
        default=Path("output/leaf_points.laz"),
        help="Output file for leaf points (default: output/leaf_points.laz)"
    )
    parser.add_argument(
        "--out-wood",
        type=Path,
        default=Path("output/wood_points.laz"),
        help="Output file for wood points (default: output/wood_points.laz)"
    )
    parser.add_argument(
        "--write-labels",
        action="store_true",
        help="If set, add PredSemantic dimension (1=leaf, 0=wood) to outputs"
    )
    args = parser.parse_args()

    leaf_on_path = args.leaf_on
    leaf_off_path = args.leaf_off
    threshold = args.threshold
    leaf_out = args.out_leaf
    wood_out = args.out_wood
    write_labels = args.write_labels

    leaf_out.parent.mkdir(parents=True, exist_ok=True)
    wood_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading leaf-on data: {leaf_on_path}")
    leaf_on_las, leaf_on_points = read_laz_raw(leaf_on_path)

    print(f"Loading leaf-off data: {leaf_off_path}")
    leaf_off_las, leaf_off_points = read_laz_raw(leaf_off_path)

    # Align in a common local frame (XY+vertical shift)
    origin = leaf_off_points.mean(axis=0)
    leaf_on_local = leaf_on_points - origin
    leaf_off_local = leaf_off_points - origin

    z_shift = np.percentile(leaf_off_local[:, 2], 5) - np.percentile(leaf_on_local[:, 2], 5)
    leaf_on_local[:, 2] += z_shift

    leaf_on_pcd = to_o3d(leaf_on_local)
    leaf_off_pcd = to_o3d(leaf_off_local)

    print("leaf-on centroid:", leaf_on_points.mean(axis=0))
    print("leaf-off centroid:", leaf_off_points.mean(axis=0))
    print("difference:", leaf_on_points.mean(axis=0) - leaf_off_points.mean(axis=0))

    print("Calculating distances (leaf-on to nearest leaf-off point)")
    dists = np.asarray(leaf_on_pcd.compute_point_cloud_distance(leaf_off_pcd))  # [web:124][web:127]

    print("distance statistics:")
    print("min:", dists.min())
    print("mean:", dists.mean())
    print("median:", np.median(dists))
    print("max:", dists.max())
    print(f"Using threshold: {threshold}")

    leaf_idx = np.where(dists > threshold)[0]
    wood_idx = np.where(dists <= threshold)[0]

    print(f"Leaf points: {leaf_idx.size}, wood points: {wood_idx.size}")

    # --- Create LAS outputs ---
    # Leaf LAS
    leaf_las = laspy.LasData(leaf_on_las.header)
    leaf_las.points = leaf_on_las.points[leaf_idx]

    # Wood LAS
    wood_las = laspy.LasData(leaf_on_las.header)
    wood_las.points = leaf_on_las.points[wood_idx]

    if write_labels:
        # Ensure PredSemantic exists in both outputs
        add_predsemantic_dimension(leaf_las)
        add_predsemantic_dimension(wood_las)

        # Set labels: 1 for leaf, 0 for wood
        leaf_las.PredSemantic = np.ones(leaf_las.points.__len__(), dtype=np.uint8)
        wood_las.PredSemantic = np.zeros(wood_las.points.__len__(), dtype=np.uint8)

    print(f"Saving leaf points to {leaf_out}")
    leaf_las.write(leaf_out)

    print(f"Saving wood points to {wood_out}")
    wood_las.write(wood_out)

    print(f"Done. Leaf: {leaf_idx.size} points, wood: {wood_idx.size} points.")


if __name__ == "__main__":
    main()