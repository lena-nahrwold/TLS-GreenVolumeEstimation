import argparse
import json
import subprocess
from pathlib import Path
from osgeo import ogr


def las_count(path: str) -> int:
    info = subprocess.run(
        ["pdal", "info", "--summary", str(path)],
        capture_output=True,
        check=True,
        text=True,
    )
    summary = json.loads(info.stdout)
    return summary["summary"]["num_points"]


def get_polygons_from_shapefile(shapefile: str) -> list[str]:
    ds = ogr.Open(shapefile)
    if ds is None:
        raise ValueError(f"Could not open shapefile: {shapefile}")

    layer = ds.GetLayer(0)
    polygons = []

    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        geom = geom.Clone()
        gtype = geom.GetGeometryType()

        if gtype == ogr.wkbPolygon:
            polygons.append(geom.ExportToWkt())

        elif gtype == ogr.wkbMultiPolygon:
            for i in range(geom.GetGeometryCount()):
                part = geom.GetGeometryRef(i)
                polygons.append(part.Clone().ExportToWkt())

    if not polygons:
        raise ValueError("No polygon geometry found in shapefile.")

    return polygons


def add_mask_field(input_shapefile: str, output_shapefile: str, field_name: str = "mask_val", value: int = 1) -> str:
    driver = ogr.GetDriverByName("ESRI Shapefile")

    input_ds = ogr.Open(input_shapefile)
    if input_ds is None:
        raise ValueError(f"Could not open shapefile: {input_shapefile}")

    input_layer = input_ds.GetLayer(0)
    srs = input_layer.GetSpatialRef()

    output_path = Path(output_shapefile)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        sidecar = output_path.with_suffix(ext)
        if sidecar.exists():
            driver.DeleteDataSource(str(output_path))
            break

    output_ds = driver.CreateDataSource(str(output_shapefile))
    output_layer = output_ds.CreateLayer(output_path.stem, srs, input_layer.GetGeomType())

    input_defn = input_layer.GetLayerDefn()
    existing_fields = [input_defn.GetFieldDefn(i).GetName() for i in range(input_defn.GetFieldCount())]

    for i in range(input_defn.GetFieldCount()):
        output_layer.CreateField(input_defn.GetFieldDefn(i))

    if field_name not in existing_fields:
        output_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))

    output_defn = output_layer.GetLayerDefn()

    for feat in input_layer:
        out_feat = ogr.Feature(output_defn)

        for i in range(input_defn.GetFieldCount()):
            field_name_in = input_defn.GetFieldDefn(i).GetName()
            out_feat.SetField(field_name_in, feat.GetField(field_name_in))

        out_feat.SetField(field_name, value)
        geom = feat.GetGeometryRef()
        if geom is not None:
            out_feat.SetGeometry(geom.Clone())

        output_layer.CreateFeature(out_feat)
        out_feat = None

    input_ds = None
    output_ds = None
    return output_shapefile


def run_pipeline(pipeline: dict) -> None:
    pipeline_json = json.dumps(pipeline)
    subprocess.run(
        ["pdal", "pipeline", "--stdin"],
        input=pipeline_json.encode("utf-8"),
        check=True,
    )


def crop_point_cloud(pointcloud: str, shapefile: str, output_path: str) -> str:
    n_in = las_count(pointcloud)
    polygons = get_polygons_from_shapefile(shapefile)

    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": pointcloud,
                "use_eb_vlr": True
            },
            {
                "type": "filters.crop",
                "polygon": polygons,
                "outside": False
            },
            {
                "type": "writers.las",
                "filename": output_path,
                "extra_dims": "all",
                "forward": "all"
            }
        ]
    }

    run_pipeline(pipeline)

    n_out = las_count(output_path)
    print("Mode: crop")
    print("Total points before:", n_in)
    print("Total points after:", n_out)
    print("Points removed:", n_in - n_out)

    return output_path


def filter_point_cloud(pointcloud: str, shapefile: str, output_path: str) -> str:
    n_in = las_count(pointcloud)

    shp_path = Path(shapefile)
    masked_shp = str(shp_path.with_name(f"{shp_path.stem}_mask.shp"))
    masked_shp = add_mask_field(shapefile, masked_shp, field_name="mask_val", value=1)

    layer_name = Path(masked_shp).stem

    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": pointcloud,
                "use_eb_vlr": True
            },
            {
                "type": "filters.assign",
                "value": "Mask = 0"
            },
            {
                "type": "filters.overlay",
                "dimension": "Mask",
                "datasource": masked_shp,
                "layer": layer_name,
                "column": "mask_val"
            },
            {
                "type": "filters.expression",
                "expression": "Mask != 1"
            },
            {
                "type": "writers.las",
                "filename": output_path,
                "extra_dims": "all",
                "forward": "all"
            }
        ]
    }

    run_pipeline(pipeline)

    n_out = las_count(output_path)
    print("Mode: filter")
    print("Total points before:", n_in)
    print("Total points after:", n_out)
    print("Points removed:", n_in - n_out)

    return output_path


def process_point_cloud(pointcloud: str, shapefile: str, output_path: str, use_as_filter: bool) -> str:
    if use_as_filter:
        return filter_point_cloud(pointcloud, shapefile, output_path)
    return crop_point_cloud(pointcloud, shapefile, output_path)


def main():
    parser = argparse.ArgumentParser(description="Crop or filter a LAS/LAZ point cloud with polygon shapefile boundaries.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input LAS/LAZ file")
    parser.add_argument("-s", "--shapefile", required=True, type=str, help="Polygon shapefile")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output LAS/LAZ file")
    parser.add_argument(
        "-f",
        "--filter",
        action="store_true",
        help="Remove points inside polygons instead of keeping them"
    )
    args = parser.parse_args()

    result = process_point_cloud(args.input, args.shapefile, args.output, args.filter)
    print(f"✓ Saved point cloud to {result}")


if __name__ == "__main__":
    main()