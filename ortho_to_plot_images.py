import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask

grid = gpd.read_file("grid_with_plots.geojson")

list_of_ortho_dates = ['13-10-2023', '31-10-2023', '14-11-2023', '22-11-2023', '05-12-2023', '12-12-2023', '19-12-2023', '12-01-2024']

for cur_ortho_date in list_of_ortho_dates:

    raster = rio.open(f'/workspaces/field pansy/orthos/FP_{cur_ortho_date}_georeferenced.png')
    gdf = grid.to_crs(raster.crs)

    for index, row in grid.iterrows():
        geom = [row.geometry]
        plot = row.Plot
        
        out_image, out_transform = mask(raster, geom, crop=True)

        # Update the metadata
        out_meta = raster.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save the cropped image
        with rio.open(f'/workspaces/field pansy/plot_images/{cur_ortho_date}_{plot}.tif', "w", **out_meta) as dest:
            dest.write(out_image)