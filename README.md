# TLS-GreenVolumeEstimation

A semi-automatic pipeline for foliage-oriented green volume estimation from Terrestrial Laser Scanning (TLS) point clouds in structurally heterogeneous, multilayered vegetation (e.g. urban forest gardens). It combines geometric semantic segmentation (RayCloudTools + Cloth Simulation Filter) with voxel-based aggregation to derive total, overstory, and understory green volume and green volume per area (GVA).

The methodology and evaluation are described in the accompanying thesis:

> Nahrwold, L. (2026). TLS-based Green Volume Estimation in a Young Temperate Urban Forest Garden. Master thesis, HNE Eberswalde.

This script orchestrates a **TLS green volume estimation pipeline** consisting of:

1. Optional tiling of input point clouds  
2. Leaf–wood (and instance) segmentation with **py-rct / RayCloudTools**  
3. Ground classification with **Cloth Simulation Filter (CSF)**  
4. Merging ground and non-ground segments into fully segmented tiles  
5. Optional cross-tile merging of smart tiles  
6. Voxel-based green volume calculation within an optional AOI shapefile

The main entry point is `run.py`.

---

## 1. Basic usage

```bash
python3 run.py \
  --input /path/to/input_dir \
  --output /path/to/output_dir \
  --shapefile /path/to/aoi.shp 
```

- `--task all` (default) runs the full pipeline (segmentation + voxelization).
- `--task segmentation` runs only segmentation steps (tiling (optional), py-rct, CSF, merging).
- `--task voxelization` runs only voxel-based green volume estimation on a segmented point cloud.

---

## 2. Top-level parameters

### Required

- `--input PATH`  
  Input LAS/LAZ **file or directory**.
  - For `--tiling basic`/`smart`, this is the raw point cloud (single file or directory of tiles).
  - For `--task voxelization`, this is expected to be a directory containing the non-ground, segmented tiles you want to voxelize.

- `--output PATH`  
  Output directory for all intermediate and final results. Subdirectories are created inside this directory.

### High-level control

- `--task {all,segmentation,voxelization}`  
  - `segmentation`: run tiling + py-rct + CSF + per-tile merging (+ optional smart merging). No voxelization.  
  - `voxelization`: run only voxel-based green volume on an existing, segmented input.  
  - `all` (default): run segmentation followed by voxelization in one go.

- `--tiling {skip,basic,smart}`  
  - `skip` (default): no tiling. Run segmentation directly on the input.  
  - `basic`: use PDAL tiling without cross-tile merging of instance IDs.  
  - `smart`: use `3dtrees_Smart_Tile` to tile and later cross-merge tree instances and semantic labels.

- `--clear-segmentation-output`  
  If set, deletes intermediate segmentation outputs (`rct_leaf_wood`, `csf_ground`) at the end, keeping only merged and voxelization results.

---

## 3. Tiling parameters

- `--tile-length INT` (default: `30`)  
  Tile size in meters.

- `--tile-buffer INT` (default: `10`)  
  Buffer overlap in meters between tiles.

- `--skip-dimension-reduction BOOL` (default: `False`)  
  Smart-tiling specific.  
  - `False`: reduce points to X,Y,Z only to reduce file size (recommended for raw pre-segmentation data).  
  - `True`: keep all dimensions (useful for post-segmentation data).

Internal paths created:

- `tiles/` inside `--output`  
- For smart tiling, py-rct input defaults to `tiles/subsampled_res1`.

---

## 4. py-rct / RayCloudTools parameters

These parameters are forwarded to `run_batch_segmentation` for leaf–wood and instance segmentation.

- `--gradient FLOAT` (default: `1.0`)  
  Gradient threshold for terrain extraction.

- `--max-diameter FLOAT` (default: `0.9`)  
  Maximum branch diameter (m) for tree instance segmentation.

- `--crop-length FLOAT` (default: `1.0`)  
  Distance from branch tip used to reconstruct QSM.

- `--distance-limit FLOAT` (default: `1.0`)  
  Maximum distance between neighbour points within a tree.

- `--height-min FLOAT` (default: `2.0`)  
  Minimum point height (m) considered to belong to a tree.

- `--girth-height-ratio FLOAT` (default: `0.12`)  
  Proportion of tree height used to estimate trunk girth.

- `--global-taper FLOAT` (default: `0.024`)  
  Global taper (diameter per unit length).

- `--global-taper-factor FLOAT` (default: `0.3`)  
  Factor scaling the global taper (0–1).

- `--gravity-factor FLOAT` (default: `0.3`)  
  Controls preference for vertical tree structures.

- `-sd`, `--split-distance FLOAT` (default: `0.02`)  
  Smaller values produce more, finer splits; larger values fewer, coarser splits.

- `--branch-segmentation`  
  If set, segmentation is per **branch**; otherwise per **tree**.

Internal segmentation output:

- `rct_leaf_wood/` inside `--output`  
  - Segmented tiles, used as input to CSF.

---

## 5. Cloth Simulation Filter (CSF) parameters

These parameters are forwarded to `run_csf` for ground classification.

- `--cloth-resolution FLOAT` (default: `0.05`)  
  Grid size of the simulated cloth. Larger values → coarser DTM; smaller values → finer terrain.

- `--rigidness {1,2,3}` (default: `2`)  
  Terrain preset:  
  - `1` = steep / rugged (soft cloth)  
  - `2` = relief (medium)  
  - `3` = flat (rigid)

- `--time-step FLOAT` (default: `0.65`)  
  Simulation time step per iteration. Larger values speed up convergence but may reduce stability.

- `--class-threshold FLOAT` (default: `0.5`)  
  Distance threshold between original points and simulated terrain for ground vs non-ground classification.

- `--iterations INT` (default: `500`)  
  Maximum number of cloth simulation iterations.

- `--slope-smooth`  
  Enable slope smoothing to improve ground detection on steep or rugged terrain.

Internal CSF output:

- `csf_ground/ground` and `csf_ground/non_ground` inside `--output`.

---

## 6. AOI & voxelization parameters

- `-s`, `--shapefile PATH` (optional)  
  AOI polygon shapefile for cropping in the voxel-based calculation. If not provided, voxelization is performed on the full spatial extent.

- `-v`, `--voxel-sizes FLOAT...` (default: `0.1 0.2 0.3`)  
  List of voxel sizes (m) used for voxelization and green volume aggregation.

Voxelization output:

- `results/` inside `--output`  
  Files written by `voxel_based_green_volume`, including green volume metrics per voxel size.

---

## 7. Internal paths and metadata

The script derives internal paths from `--output`:

- `tiles/`  
- `rct_leaf_wood/`  
- `csf_ground/`  
- `results/segmented_laz/` (per-tile merged segmentation)  
- `results/` (voxelization results)

A metadata file is written at the start of segmentation runs:

- `run_metadata.json` in `--output`  
  Contains all effective parameters, command-line arguments, and timestamps.

---

## 8. Example workflow: segmentation → manual correction → voxelization

This section describes a recommended two-stage workflow:

1. Run segmentation once.  
2. Manually inspect and edit segmentation results.  
3. Run voxelization on the corrected data.

### 8.1 Stage 1: run segmentation only

Run the pipeline with `--task segmentation` to produce segmented tiles and **stop before voxelization**.

```bash
python3 run.py \
  --input /path/to/raw_las_laz \
  --output /path/to/output_seg \
  --task segmentation \
  --tiling \
  --tile-length 30 \
  --tile-buffer 10 \
  --crop-length 0.1 \
  --cloth-resolution 0.5 \
  --class-threshold 0.02
```

This will:

1. Optionally tile the input (`tiles/` under `output_seg`).  
2. Run py-rct segmentation into `output_seg/rct_leaf_wood/`.  
3. Run CSF ground classification into `output_seg/csf_ground/ground` and `output_seg/csf_ground/non_ground`.  
4. Merge ground and non-ground tiles into per-tile, fully segmented point clouds in:  
   - `output_seg/results/segmented_laz/`  
5. For `--tiling smart`, run cross-tile merging (ID and semantic merging).

After this step, you have a set of segmented LAS/LAZ files ready for manual QA.

### 8.2 Stage 2: manual corrections

Use your preferred point cloud editor (e.g. CloudCompare, other tools) to:

- Inspect files in `output_seg/results/segmented_laz/`.  
- Correct semantic labels, tree instance IDs, or remove artefacts as needed.  
- Save the corrected tiles to a new directory, e.g.:

```text
/path/to/corrected_segmentation/
  tile_001_corrected.laz
  tile_002_corrected.laz
  ...
```

Important:

- Ensure the corrected files retain the expected semantic dimension names and formats (e.g. `PredSemantic` if required by downstream code).  
- Ensure all tiles you want to include in voxelization are in a *single* directory.

### 8.3 Stage 3: run voxelization only

Once corrections are done, run the voxel-based green volume step with `--task voxelization`, using the **corrected** directory as `--input`.

```bash
python3 run.py \
  --input /path/to/corrected_segmentation \
  --output /path/to/output_vox \
  --task voxelization \
  --voxel-sizes 0.1 0.2 0.3 \
  --shapefile /path/to/aoi.shp
```

In this mode:

- Tiling, py-rct, and CSF are **skipped**.  
- `input_path` passed to `voxel_based_green_volume` is the directory given by `--input`.  
- Results are written into `output_vox/results/`.

You can repeat Stage 3 with different voxel sizes without re-running segmentation:

```bash
python3 run.py \
  --input /path/to/corrected_segmentation \
  --output /path/to/output_vox_05_10 \
  --task voxelization \
  --voxel-sizes 0.05 0.10 \
  --shapefile /path/to/aoi.shp
```

---

## 9. Example: full end-to-end run (no manual step)

If you do not need manual correction and want an end-to-end run in one go:

```bash
python3 run.py \
  --input /path/to/raw_las_laz \
  --output /path/to/output_full \
  --task all \
  --tiling \
  --tile-length 30 \
  --tile-buffer 10 \
  --gradient 1.0 \
  --max-diameter 0.9 \
  --cloth-resolution 0.05 \
  --class-threshold 0.5 \
  --voxel-sizes 0.1 0.2 0.3 \
  --shapefile /path/to/aoi.shp
```

This will run:

1. Tiling (if selected).  
2. py-rct segmentation.  
3. CSF ground classification.  
4. Per-tile merging (+ optional cross-tile merging).  
5. Voxel-based green volume, using the internal CSF non-ground directory.

---

## 10. Optional pre‑processing: crop point cloud to AOI or filter by polygon

Before running the TLS Green Volume Pipeline, you can optionally **crop the raw point cloud to the AOI** using a polygon shapefile. This reduces file size and processing time, and ensures that all subsequent steps operate only within the area of interest.

The script `crop_point_cloud.py` provides a lightweight PDAL-based pre-processing step to clip TLS point clouds with a polygon shapefile. It can be used in two modes:

- **Crop mode (default)** – keep only points **inside** the AOI polygon.  
- **Filter mode (`--filter`)** – use the shapefile as a mask and keep only points **outside** the polygon.

Basic usage:

```bash
python3 crop_point_cloud.py \
  --input /path/to/raw_las_laz \
  --output /path/to/output/folder \
  --shapefile /path/to/aoi.shp
```

### Local Installation
To setup the pipeline with a local installation, clone this repository, install RayCloudTools (and TreeTools) following the instructions in the [RayCloudTools repository](https://github.com/csiro-robotics/raycloudtools) and run:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker usage

### Build the Docker image

From the repository root:

```bash
sudo docker build -t tls-green-volume -f docker/Dockerfile .
```
Adjust the `run_docker_pipeline.sh`, `run_docker_segmentation.sh`, `run_docker_voxelization.sh` scripts.


## References

> Lowe, Thomas, and Kazys Stepanas. "RayCloudTools: A Concise Interface for Analysis and Manipulation of Ray Clouds." IEEE Access (2021).

> Devereux, T., Lowe, T., Rivory, J., Reckziegel, R. B., Calders, K., Aryal, R. R., Eaton, G., Cooper, Z., Levick, S., Phinn, S., & Woodgate, W. (2026). RayExtract: A fast, scalable method for tree volume reconstruction from terrestrial laser scanning. Remote Sensing of Environment, 334, 115162. https://doi.org/10.1016/j.rse.2025.115162

> Zhang, W., Qi, J., Peng, W., Wang, H., Xie, D., Wang, X., and Yan, G. (2016). An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation. Remote Sensing, 8:501.