import argparse
import os
import laspy
import json
import numpy as np
from estimate_area_from_shp import estimate_area_from_shp


def compute_voxel_volume(coords, voxel_sizes):
    volumes = []

    # shift to local origin
    coords_min = coords.min(axis=0)
    coords_local = coords - coords_min

    for voxel_size in voxel_sizes:
        vox = np.floor(coords_local / voxel_size).astype(np.int64)

        n_vox = np.unique(vox, axis=0).shape[0]
        volume = n_vox * voxel_size**3

        volumes.append(volume)

    volumes = np.array(volumes)

    mean_volume = volumes.mean()
    std_volume = volumes.std(ddof=1)
    cv = std_volume / mean_volume if mean_volume != 0 else 0

    return volumes, mean_volume, std_volume, cv


def get_coords(las, mask=None):
    return np.column_stack((las.x[mask], las.y[mask], las.z[mask]))


def voxel_based_green_volume(
        input_path: str = None,
        output_dir: str = None,
        voxel_sizes: list = [0.1, 0.2, 0.3],
        class_labels: list = None,
        area_size: float = None
    ) -> str:

    # Resolve input
    if os.path.isfile(input_path):
        files = [input_path]

    elif os.path.isdir(input_path):
        files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".las", ".laz"))
        ])

        if not files:
            raise ValueError("No LAS/LAZ files found in directory.")

    else:
        raise ValueError("Input path is neither a file nor a directory.")

    print(f"Found {len(files)} file(s) for processing.")

    keys = ["Total", "Crowns", "Low Vegetation"]

    # aggregation container
    aggregated = {
        key: np.zeros(len(voxel_sizes), dtype=float)
        for key in keys
    }

    # Process each file
    for f in files:
        print(f"Processing: {os.path.basename(f)}")

        las = laspy.read(f)

        standard_dims = set(las.point_format.dimension_names)
        extra_dims = set(las.point_format.extra_dimension_names)

        labels = None
        for name in ('Label', 'label', 'PredSemantic', 'classification'):
            if name in standard_dims or name in extra_dims:
                labels = np.asarray(las[name])
                break
            
        if labels is None:
            raise ValueError("No label field found")

        crown_label = class_labels[0]
        lowveg_label = class_labels[1] if len(class_labels) > 1 else None

        mask_total = np.isin(labels, [crown_label] + ([lowveg_label] if lowveg_label is not None else []))
        mask_crown = labels == crown_label

        if lowveg_label is not None:
            mask_lowveg = labels == lowveg_label
        else:
            mask_lowveg = None

        results = {
        "Total": compute_voxel_volume(get_coords(las, mask_total), voxel_sizes),
        "Crowns": compute_voxel_volume(get_coords(las, mask_crown), voxel_sizes),
        }

        if mask_lowveg is not None and np.any(mask_lowveg):
            results["Low Vegetation"] = compute_voxel_volume(get_coords(las, mask_lowveg), voxel_sizes)
        else:
            # Initialize empty zero volumes if no low vegetation
            results["Low Vegetation"] = (
                np.zeros(len(voxel_sizes)), 0.0, 0.0, 0.0
            )

        # Accumulate volumes
        for key in keys:
            volumes, *_ = results[key]
            aggregated[key] += volumes


    # Output file
    output_file = os.path.join(output_dir, "green_volume.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write results
    lines = []
    lines.append("===== Green Volume Estimates =====")
    lines.append(f"Processed files: {len(files)}")
    for f in files:
        lines.append(f"- {os.path.basename(f)}")

    lines.append("")

    for key in keys:
        volumes = aggregated[key]

        mean_volume = volumes.mean()
        std_volume = volumes.std(ddof=1)
        cv = std_volume / mean_volume if mean_volume != 0 else 0

        lines.append(f"===== {key.upper()} =====")

        for v, vol in zip(voxel_sizes, volumes):
            vpa = vol / area_size if area_size else None
            vpa_str = f"{vpa:.3f}" if vpa is not None else "N/A"

            lines.append(
                f"Voxel size = {v} m → Volume = {vol:.6f} m³ → Volume per area = {vpa_str} m³/m²"
            )

        lines.append("----- Summary over voxel sizes -----")
        lines.append(f"Mean volume = {mean_volume:.6f} m³")
        lines.append(f"Std dev     = {std_volume:.6f} m³")
        lines.append(f"CV          = {cv:.4f} ({cv*100:.2f} %)")

        lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Voxelize point cloud and compute volume")

    parser.add_argument(
        "-i", "--input", required=True, type=str,
        help="Path to directory or single LAS/LAZ file."
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str,
        help="Output directory."
    )
    parser.add_argument(
        "-v", "--voxel-size", nargs="+", default=[0.1, 0.2, 0.3], type=float,
        help="Voxel sizes."
    )
    parser.add_argument(
        "-l", "--class-label", nargs="+", default=[2, 3], type=int,
        help="Class labels: [overstory, understory]"
    )
    parser.add_argument(
        "-s", "--area-size", type=float,
        help="Area size for normalization (m²)."
    )
    parser.add_argument("--shapefile", type=str,
                    help="Shapefile used for cropping the point cloud to AOI. Used for calculating the area size.")

    args = parser.parse_args()

    if args.shapefile:
        area_size = estimate_area_from_shp(args.shapefile)
    elif args.area_size:
        area_size = args.area_size
    else:
        area_size = None
        print("No area size or path to shapefile for area size estimation provided.")

    voxel_based_green_volume(
        input_path=args.input,
        output_dir=args.output,
        voxel_sizes=args.voxel_size,
        class_labels=np.array(args.class_label),
        area_size=area_size
    )


if __name__ == '__main__':
    main()