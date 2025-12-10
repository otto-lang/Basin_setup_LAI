This code creates a new topo.nc file for use in iSnobal from an input:
- Shapefile (.shp)
- DEM (.tif)
- Veg height raster (.tif)
- LAI image (.tif)
- NLCD Land cover raster (.tif)

Therby acting as an alternative to basin_setup for iSnobal

In the config.yaml file, the user can define the output coordinate system, resolution, resampling techniques, and choose which NLCD land cover type the LAI image should be used to derive veg_k and veg_tau, the canopy extinction and 
transmissivity coefficients.
 
In its current implementation:
- Veg height is updated using the CONUS GEDI veg height .tif available here: https://glad.umd.edu/dataset/gedi/
- Veg type is simplified to the latest CONUS NCLD Veg Land cover, available here: https://www.mrlc.gov/data?f%5B0%5D=project_tax_term_term_parents_tax_term_name%3AAnnual%20NLCD
- Veg_k and veg_tau are calculated predominantly from Sentinel-2 SNAP LAI scenes. LAI scenes are freely and easily available for download here:
       https://custom-scripts.sentinel-hub.com/sentinel-2/lai/ by clicking on "Copernicus Browser". This will enable a mapping interface where you can easily 
       toggle dates to select cloud-free snow-free LAI scenes for your region. Once you have selected your scene, you can select a bounding box and download the scene
       as an analytical geotiff file. You will have to log in or create a free account to do the analytical download step. 

Notes on LAI-based veg_k and veg_tau:
- Images must have no snow and no clouds. Ideally, we want LAI to be measured when deciduous cover has dropped all its leaves as well. Finding a snow-free, leaf-free scene is often challenging or impractical if we want the most recent vegetation,
    Therefore, I am only using this LAI approach for evergreen vegetation (classified by the latest NLCD land cover). Deciduous and mixed vegetation retain their default k and tau using this approach (USE_FIXED_DECID_MIXED = True). 
    Seems to work well in western mountain watersheds, where there is little deciduous cover. This approach is less ideal in deciduous forest basins (ex. East Coast), where a better approach would be to take care in selecting a leaf and snow-free 
    images, and use LAI-derived k and tau for deciduous and mixed cover as well (set USE_FIXED_DECID_MIXED = False). 

After downloading your input data for the region of interest, update the paths in the config.yaml file and run the code with these commands:
```bash
conda env create -f environment.yaml
conda activate new_setup
python create_topo.py
```
