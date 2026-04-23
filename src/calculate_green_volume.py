import argparse
import os

import laspy
import numpy as np
from tqdm import tqdm

from estimate_area_from_shp import estimate_area_from_shp


def update_voxel_sets(coords, voxel_sizes, voxel_sets, origin):
    if coords.size == 0:
        return

    coords_local = coords - origin

    for idx, voxel_size in enumerate(voxel_sizes):
        vox = np.floor(coords_local / voxel_size).astype(np.int64)
        voxel_sets[idx].update(map(tuple, vox))


def summarize_volumes(volumes):
    mean_volume = volumes.mean() if len(volumes) > 0 else 0.0
    std_volume = volumes.std(ddof=1) if len(volumes) > 1 else 0.0
    cv = std_volume / mean_volume if mean_volume != 0 else 0.0
    return mean_volume, std_volume, cv


def voxel_based_green_volume(
    input_path: str = None,
    output_dir: str = None,
    voxel_sizes: list = [0.1, 0.2, 0.3],
    class_labels: list = None,
    dimension: str = "PredSemantic",
    area_size: float = None,
    shapefile: str = None,
    chunk_size: int = 2_000_000,
) -> str:
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

    if shapefile:
        area_size = estimate_area_from_shp(shapefile)
        print(f"Estimated area from shapefile: {area_size:.3f} m²")

    keys = ["Total", "Crowns", "Low Vegetation"]

    aggregated_sets = {
        key: [set() for _ in voxel_sizes]
        for key in keys
    }

    crown_label = class_labels[0]
    lowveg_label = class_labels[1] if len(class_labels) > 1 else None

    for file_idx, f in enumerate(files, start=1):
        basename = os.path.basename(f)
        print(f"[{file_idx}/{len(files)}] Processing: {basename}")

        with laspy.open(f) as reader:
            total_points = reader.header.point_count
            origin = np.array(
                [reader.header.x_min, reader.header.y_min, reader.header.z_min],
                dtype=np.float64
            )

            point_format = reader.header.point_format
            standard_dims = set(point_format.dimension_names)
            extra_dims = set(point_format.extra_dimension_names)

            if dimension not in standard_dims and dimension not in extra_dims:
                raise ValueError(f"Given dimension '{dimension}' not found in {f}.")

            chunk_counter = 0

            with tqdm(
                total=total_points,
                desc=basename,
                unit="pts",
                unit_scale=True,
            ) as pbar:
                for points in reader.chunk_iterator(chunk_size):
                    chunk_counter += 1

                    labels = np.asarray(points[dimension])

                    x = points.x
                    y = points.y
                    z = points.z

                    mask_crown = labels == crown_label
                    mask_lowveg = labels == lowveg_label if lowveg_label is not None else None
                    mask_total = np.isin(
                        labels,
                        [crown_label] + ([lowveg_label] if lowveg_label is not None else [])
                    )

                    if np.any(mask_total):
                        coords_total = np.column_stack((x[mask_total], y[mask_total], z[mask_total]))
                        update_voxel_sets(coords_total, voxel_sizes, aggregated_sets["Total"], origin)

                    if np.any(mask_crown):
                        coords_crown = np.column_stack((x[mask_crown], y[mask_crown], z[mask_crown]))
                        update_voxel_sets(coords_crown, voxel_sizes, aggregated_sets["Crowns"], origin)

                    if mask_lowveg is not None and np.any(mask_lowveg):
                        coords_lowveg = np.column_stack((x[mask_lowveg], y[mask_lowveg], z[mask_lowveg]))
                        update_voxel_sets(coords_lowveg, voxel_sizes, aggregated_sets["Low Vegetation"], origin)

                    pbar.update(len(points))

                    if chunk_counter % 10 == 0:
                        postfix = {
                            "chunks": chunk_counter,
                            "total@v1": len(aggregated_sets["Total"][0]),
                            "crowns@v1": len(aggregated_sets["Crowns"][0]),
                        }

                        if lowveg_label is not None:
                            postfix["lowveg@v1"] = len(aggregated_sets["Low Vegetation"][0])

                        pbar.set_postfix(postfix, refresh=False)

            print(f"  Finished {chunk_counter} chunk(s).")

    aggregated = {
        key: np.array(
            [len(s) * (v ** 3) for s, v in zip(aggregated_sets[key], voxel_sizes)],
            dtype=float
        )
        for key in keys
    }

    output_file = os.path.join(output_dir, "green_volume.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = []
    lines.append("===== Green Volume Estimates =====")
    lines.append(f"Processed files: {len(files)}")
    for f in files:
        lines.append(f"- {os.path.basename(f)}")
    lines.append("")

    for key in keys:
        volumes = aggregated[key]
        mean_volume, std_volume, cv = summarize_volumes(volumes)

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
    
    result_text = "\n".join(lines)

    print("\n" + result_text)

    with open(output_file, "w") as f:
        f.write(result_text)

    print(f"Done. Results written to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Voxelize point cloud and compute green volume")

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
        "-d", "--dimension", default="PredSemantic", type=str,
        help="Dimension name for semantic class labels, e.g. 'PredSemantic' or 'Classification'."
    )
    parser.add_argument(
        "-l", "--class-label", nargs="+", default=[2, 3], type=int,
        help="Class labels: [overstory, understory]"
    )
    parser.add_argument(
        "-s", "--area-size", type=float,
        help="Area size for normalization (m²)."
    )
    parser.add_argument(
        "--shapefile", type=str,
        help="Shapefile used for cropping the point cloud to AOI. Used for calculating the area size."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2_000_000,
        help="Number of points to read per chunk."
    )

    args = parser.parse_args()

    voxel_based_green_volume(
        input_path=args.input,
        output_dir=args.output,
        voxel_sizes=args.voxel_size,
        dimension=args.dimension,
        class_labels=np.array(args.class_label),
        area_size=args.area_size if args.area_size else None,
        shapefile=args.shapefile if args.shapefile else None,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()