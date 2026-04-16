import argparse
import os
import json
import subprocess
import numpy as np
from osgeo import ogr

def crop_point_cloud(pointcloud:str, shapefile:str, output_path:str, use_as_filter:bool) -> str:

    ds = ogr.Open(shapefile)
    layer = ds.GetLayer(0)
    feat = layer.GetNextFeature()
    geom = feat.GetGeometryRef()
    wkt = geom.ExportToWkt()

    pipeline = {
        "pipeline": [
            pointcloud, 
            {
                "type": "filters.crop",
                "polygon": wkt,
                "outside": use_as_filter,
            },
            output_path
        ]
    }

    pipeline_json = json.dumps(pipeline)

    subprocess.run(
        ["pdal", "pipeline", "--stdin"],
        input=pipeline_json.encode("utf-8"),
        check=True,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("-i","--input", required=True, type=str)
    parser.add_argument("-s","--shapefile", required=True, type=str)
    parser.add_argument("-o","--output", required=True, type=str)
    parser.add_argument("-f", "--filter", default=False)
    args = parser.parse_args()

    pcd = args.input
    shp = args.shapefile
    output = args.output
    use_as_filter = args.filter

    cropped_point_cloud = crop_point_cloud(pcd, shp, output, use_as_filter)

    print(f"✓ Saved cropped point cloud to {cropped_point_cloud}.")


if __name__ == '__main__':
    main()