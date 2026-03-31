import geopandas as gpd
import argparse

def estimate_area_from_shp(shapefile:str) -> float:
    # Read shapefile
    gdf = gpd.read_file(shapefile)
    print(gdf.crs) 

    area = gdf.geometry.area

    print(f"Area size = {area} m²")

    return area

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("-s","--shapefile", required=True, type=str)
    args = parser.parse_args()

    area = estimate_area_from_shp(args.shapefile)


if __name__ == '__main__':
    main()