import argparse
import os
import pdal
import json
import numpy as np

def compute_voxel_volume(coords, voxel_sizes, area_size=None):
    volumes = []
    volumes_per_area = []

    for voxel_size in voxel_sizes:
        vox = np.floor(coords / voxel_size).astype(np.int64)
        n_vox = np.unique(vox, axis=0).shape[0]

        volume = n_vox * voxel_size**3
        volumes.append(volume)

        if area_size:
            volumes_per_area.append(volume / area_size)

    volumes = np.array(volumes)

    mean_volume = volumes.mean()
    std_volume = volumes.std(ddof=1)
    cv = std_volume / mean_volume

    return volumes, volumes_per_area, mean_volume, std_volume, cv

def get_coords(arrays, mask):
    return np.stack((
        arrays["X"][mask],
        arrays["Y"][mask],
        arrays["Z"][mask]
    ), axis=1).astype(np.float32)

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

    # Prepare aggregation
    keys = ["Total", "Crowns", "Low Vegetation"]
    aggregated = {k: np.zeros(len(voxel_sizes)) for k in keys}

    # Process each file 
    for f in files:
        print(f"Processing: {os.path.basename(f)}")

        # Load point cloud
        pipeline = pdal.Pipeline(json.dumps([{
            "type": "readers.las",
            "filename": f
        }]))

        pipeline.execute()
        arrays = pipeline.arrays[0]

        # Find label field
        for name in ('Label', 'label', 'PredSemantic', 'Classification', 'classification'):
            if name in arrays.dtype.names:
                label_field = name
                break
        else:
            raise ValueError(f"No label field found in {f}")

        labels = arrays[label_field]

        # Define labels
        crown_label = class_labels[0]
        lowveg_label = class_labels[1]

        mask_total = np.isin(labels, [crown_label, lowveg_label])
        mask_crown = labels == crown_label
        mask_lowveg = labels == lowveg_label

        coords_total = get_coords(arrays, mask_total)
        coords_crown = get_coords(arrays, mask_crown)
        coords_lowveg = get_coords(arrays, mask_lowveg)

        results = {
            "Total": compute_voxel_volume(coords_total, voxel_sizes, area_size),
            "Crowns": compute_voxel_volume(coords_crown, voxel_sizes, area_size),
            "Low Vegetation": compute_voxel_volume(coords_lowveg, voxel_sizes, area_size),
        }

        # Aggregate volumes
        for key in keys:
            volumes, *_ = results[key]
            aggregated[key] += volumes

    output_file = os.path.join(
        f"{output_dir}",
        f"green_volume.txt"
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write results
    lines = []
    lines.append(f"===== Green Volume Estimates =====")
    lines.append(f"Processed files: {len(files)}")
    for f in files:
        lines.append(f"- {os.path.basename(f)}")

    for key, volumes in aggregated.items():
        lines.append(f"===== {key.upper()} =====")

        for v, vol in zip(voxel_sizes, volumes):
            vpa = vol / area_size if area_size else None
            vpa_str = f"{vpa:.3f}" if vpa is not None else "N/A"

            lines.append(
                f"Voxel size = {v} m → Volume = {vol:.6f} m³ → Volume per area = {vpa_str} m³/m²"
            )

        lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return output_file

def main():
    parser = argparse.ArgumentParser(description="Voxelize point cloud and compute volume")
    parser.add_argument("-i","--input", required=True, type=str,
                        help="Path to directory containing segmented LAZ point cloud(s) with semantic class labels.")
    parser.add_argument("-o","--output", required=True, type=str,
                        help="Output directory.")
    parser.add_argument("-v","--voxel-size", nargs="+", default=[0.1,0.2,0.3], type=float,
                        help="List of voxel sizes used for voxelization of the point cloud.")
    parser.add_argument("-l","--class-label", nargs="+", default=[2, 3], type=int,
                        help="List of semantic classes. The first one is used as overstory class, the second as understory. Default classes are 2 (crown) and 3 (low vegetation).")
    parser.add_argument("-s","--area-size", type=float,
                        help="Needed for green volume per area (m³/m²) estimation.")
    args = parser.parse_args()

    input_path = args.point_cloud
    output_dir = args.output
    voxel_sizes = args.voxel_size
    class_labels = np.array(args.class_label)
    area_size = args.area_size

    green_volume = voxel_based_green_volume(input_path, output_dir, voxel_sizes, class_labels, area_size)


if __name__ == '__main__':
    main()