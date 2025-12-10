"""
Build SMRF/AWSM-style topo.nc from DEM, shapefile, veg height (GEDI),
Sentinel-2 LAI, and NLCD land cover, using LAI-derived veg_k and veg_tau.
"""

import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.features import rasterize
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from datetime import datetime
from pyproj import CRS as PJCRS
import yaml  # NEW: for reading config.yaml

# --------------------------- LOAD CONFIG ---------------------------
# -------------------------------------------------------------------

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Inputs, extent, resolution, coordinate system settings
CRS_EPSG       = cfg["crs_epsg"]
DEM_PATH       = cfg["dem_path"]
SHAPE_PATH     = cfg["shape_path"]
VEG_H_TIF_PATH = cfg["veg_h_tif_path"]
LAI_PATH       = cfg["lai_path"]
NLCD_TIF_PATH  = cfg["nlcd_tif_path"]

SAVE_OUTPUT    = bool(cfg.get("save_output", True))
OUT_NC_PATH    = cfg.get("out_nc_path", "New_topo.nc")
TARGET_RES_M   = float(cfg.get("target_res_m", 100.0))
BBOX_EXPAND_PCT = float(cfg.get("bbox_expand_pct", 0.0))

# Resampling choices
RESAMPLE_CONTINUOUS  = cfg.get("resample_continuous", "bilinear")
RESAMPLE_CATEGORICAL = cfg.get("resample_categorical", "nearest")

# QC / limits
VH_MAX_TO_KEEP = float(cfg.get("vh_max_to_keep", 25.0))
LAI_SCALE      = float(cfg.get("lai_scale", 3.0))
LAI_FLOOR      = float(cfg.get("lai_floor", 0.001))

# Veg k & tau settings
ZERO_TAU_TYPES = cfg.get(
    "zero_tau_types",
    [11, 12, 21, 22, 23, 24, 31, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 92],
)
USE_FIXED_DECID_MIXED = bool(cfg.get("use_fixed_decid_mixed", True))

DECIDUOUS_41_k   = float(cfg.get("deciduous_41_k", 0.025))
DECIDUOUS_41_tau = float(cfg.get("deciduous_41_tau", 0.44))
MIXED_43_k       = float(cfg.get("mixed_43_k", 0.033))
MIXED_43_tau     = float(cfg.get("mixed_43_tau", 0.30))


# -------------------------------------------------------------------
# -------------------------------------------------------------------

def _resampling(name: str) -> Resampling:
    return {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear}[name]


def ensure_crs_and_reproject(da: xr.DataArray, crs_epsg: str, continuous=True) -> xr.DataArray:
    """Force a DataArray to the desired CRS using appropriate resampling."""
    if da.rio.crs is None:
        da = da.rio.write_crs(crs_epsg, inplace=False)
    elif str(da.rio.crs.to_epsg()) != crs_epsg.replace("EPSG:", ""):
        da = da.rio.reproject(
            crs_epsg,
            resampling=(Resampling.bilinear if continuous else Resampling.nearest),
        )
    return da


def build_projection_var(tmpl_da: xr.DataArray, crs_epsg: str) -> xr.DataArray:

    crs = PJCRS.from_user_input(crs_epsg)

    # ---- WKT1_GDAL so spatialnc finds 'SPHEROID' ----
    try:
        wkt = crs.to_wkt(version="WKT1_GDAL")
    except TypeError:
        wkt = crs.to_wkt("WKT1_GDAL")

    proj_name = crs.name or "projected CRS"
    geog = crs.geodetic_crs
    geog_name = geog.name if geog else "unknown"
    ellps = geog.ellipsoid if geog else None

    # Extract TM params (pyproj names may vary across versions)
    params = {
        p.name: p.value
        for p in (crs.coordinate_operation.params if crs.coordinate_operation else [])
    }

    # Detect UTM zone & hemisphere from EPSG code
    zone = None
    hemi_n = True
    try:
        epsg_code = int(crs.to_epsg() or 0)
        if 32601 <= epsg_code <= 32660:
            zone = epsg_code - 32600
            hemi_n = True
        elif 32701 <= epsg_code <= 32760:
            zone = epsg_code - 32700
            hemi_n = False
    except Exception:
        pass

    # CF parameter values
    lat0 = float(
        params.get(
            "Latitude of natural origin",
            params.get("latitude_of_origin", 0.0),
        )
    )
    lon0_default = (-183.0 + 6.0 * zone) if zone else 0.0
    lon0 = float(
        params.get(
            "Longitude of natural origin",
            params.get(
                "longitude_of_origin",
                params.get("central_meridian", lon0_default),
            ),
        )
    )
    k0 = float(
        params.get(
            "Scale factor at natural origin",
            params.get(
                "scale_factor_at_natural_origin",
                params.get("scale_factor", 0.9996),
            ),
        )
    )
    fe = float(params.get("False easting", 500000.0))
    fn = float(params.get("False northing", 0.0 if hemi_n else 10000000.0))

    # GeoTransform string from affine (c a b f d e)
    T = tmpl_da.rio.transform()
    geotransform_str = f"{T.c} {T.a} {T.b} {T.f} {T.d} {T.e}"

    # Ellipsoid parameters
    semi_major = float(ellps.semi_major_metre) if ellps else 6378137.0
    inv_flat = float(ellps.inverse_flattening) if ellps else 298.257223563
    semi_minor = semi_major * (1.0 - 1.0 / inv_flat)

    # Scalar int64 with EPSG code as value
    val = np.int64(int(crs.to_epsg() or 0))
    proj = xr.DataArray(val, name="projection")

    # CF + GDAL-ish + convenience attrs
    proj.attrs.update(
        {
            "grid_mapping_name": "transverse_mercator",
            "projected_crs_name": proj_name,
            "geographic_crs_name": geog_name,
            "horizontal_datum_name": "World Geodetic System 1984",
            "reference_ellipsoid_name": "WGS 84",
            "prime_meridian_name": "Greenwich",
            "longitude_of_prime_meridian": 0.0,
            "latitude_of_projection_origin": lat0,
            "longitude_of_central_meridian": lon0,
            "scale_factor_at_central_meridian": k0,
            "false_easting": fe,
            "false_northing": fn,
            "semi_major_axis": semi_major,
            "semi_minor_axis": semi_minor,
            "inverse_flattening": inv_flat,
            # WKT1 for spatialnc
            "crs_wkt": wkt,
            "spatial_ref": wkt,
            # GDAL GeoTransform
            "GeoTransform": geotransform_str,
        }
    )

    if zone is not None:
        proj.attrs["utm_zone_number"] = np.int64(zone)

    return proj


# -------------------------- PREP GRID FROM DEM ---------------------

# Load DEM as template grid
dem_src = rxr.open_rasterio(DEM_PATH, masked=True)
if "band" in dem_src.dims:
    dem_src = dem_src.squeeze("band", drop=True)

# If DEM lacks a CRS, assign specified EPSG; if it's not that EPSG, reproject it.
dem_src = ensure_crs_and_reproject(dem_src, CRS_EPSG, continuous=True)

# Load shapefile and reproject to DEM CRS
gdf = gpd.read_file(SHAPE_PATH)
if gdf.crs is None:
    raise RuntimeError("Shapefile has no CRS.")
gdf = gdf.to_crs(dem_src.rio.crs)

# Core rectangular extent (exact shapefile bbox)
minx, miny, maxx, maxy = gdf.total_bounds
core_rect = box(minx, miny, maxx, maxy)

# Expanded rectangular extent by configured percent (total)
width = maxx - minx
height = maxy - miny
half_expand_x = 0.5 * BBOX_EXPAND_PCT * width
half_expand_y = 0.5 * BBOX_EXPAND_PCT * height
expanded_rect = box(
    minx - half_expand_x,
    miny - half_expand_y,
    maxx + half_expand_x,
    maxy + half_expand_y,
)

# Clip DEM to the expanded rectangular bbox, then mask to polygon
dem_clip_bbox = dem_src.rio.clip_box(*expanded_rect.bounds)
dem_clip = dem_clip_bbox.rio.clip(
    [expanded_rect.__geo_interface__],
    gdf.crs,
    drop=True,
    invert=False,
)
dem_clip = dem_clip.astype("float32")

# Reproject DEM to the requested output resolution (same CRS; resample to new grid spacing)
dem_grid = dem_clip.rio.reproject(
    dem_clip.rio.crs,
    resolution=TARGET_RES_M,
    resampling=Resampling.bilinear,
)

# Rasterize masks on the *final* grid
transform = dem_grid.rio.transform()
out_shape = (dem_grid.sizes["y"], dem_grid.sizes["x"])

# Expanded mask: 1 inside expanded rectangle (used ONLY to mask DEM), else 0
expanded_mask = rasterize(
    [(expanded_rect, 1)],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype="uint8",
)
mask_expanded_da = xr.DataArray(
    expanded_mask,
    coords={"y": dem_grid.y, "x": dem_grid.x},
    dims=("y", "x"),
).astype("uint8")

# Core mask (requested output "mask"): 1 inside *core* rectangle, 0 in the buffer ring & outside
core_mask = rasterize(
    [(core_rect, 1)],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype="uint8",
)
mask_core_da = xr.DataArray(
    core_mask,
    coords={"y": dem_grid.y, "x": dem_grid.x},
    dims=("y", "x"),
).astype("uint8")

# Keep rioxarray metadata by masking the final DEM grid with the expanded mask (unchanged logic for DEM masking)
dem = dem_grid.where(mask_expanded_da == 1)
dem = dem.astype("float32")
dem.name = "dem"
dem.attrs.update({"units": "m", "long_name": "Elevation"})

tmpl = dem  # template grid (user-defined resolution in desired EPSG)

# ---------------------- VEG HEIGHT -> topo grid --------------------

vh = rxr.open_rasterio(VEG_H_TIF_PATH, masked=True)
if "band" in vh.dims:
    vh = vh.squeeze("band", drop=True)
vh = ensure_crs_and_reproject(vh, CRS_EPSG, continuous=True)
vh_on_topo = vh.rio.reproject_match(
    tmpl,
    resampling=_resampling(RESAMPLE_CONTINUOUS),
)
vh_on_topo = xr.where(vh_on_topo > VH_MAX_TO_KEEP, 0, vh_on_topo).fillna(0)
vh_on_topo.name = "veg_height"
vh_on_topo = vh_on_topo.astype("float32")
vh_on_topo.attrs.setdefault("units", "m")
vh_on_topo.attrs.setdefault("long_name", "Vegetation height")

# -------------------------- LAI -> topo grid -----------------------

lai_src = rxr.open_rasterio(LAI_PATH, masked=True)
if "band" in lai_src.dims:
    lai_src = lai_src.squeeze("band", drop=True)
lai_src = ensure_crs_and_reproject(lai_src, CRS_EPSG, continuous=True)
lai_on_topo = lai_src.rio.reproject_match(
    tmpl,
    resampling=_resampling(RESAMPLE_CONTINUOUS),
)
lai_on_topo = (lai_on_topo * LAI_SCALE).fillna(0)
lai_on_topo.name = "lai"

# -------------------------- NLCD -> veg_type -----------------------

nlcd = rxr.open_rasterio(NLCD_TIF_PATH, masked=True, chunks=True)
if "band" in nlcd.dims:
    nlcd = nlcd.squeeze("band", drop=True)
if nlcd.rio.crs is None:
    raise RuntimeError("NLCD GeoTIFF has no CRS.")
nlcd = ensure_crs_and_reproject(nlcd, CRS_EPSG, continuous=False)

src_nodata = nlcd.rio.nodata
if src_nodata is None:
    nlcd = nlcd.rio.write_nodata(0, inplace=False)

nlcd_on_topo = nlcd.rio.reproject_match(
    tmpl,
    resampling=_resampling(RESAMPLE_CATEGORICAL),
    nodata=0,
).assign_coords(x=tmpl.x, y=tmpl.y)
nlcd_on_topo = nlcd_on_topo.reset_coords(drop=True).fillna(0).astype("uint16")
nlcd_on_topo.name = "veg_type"
nlcd_on_topo.attrs.update(
    {
        "long_name": "Vegetation type (NLCD 2024 classes)",
        "source": Path(NLCD_TIF_PATH).name,
        "category_encoding": "NLCD 2024 class codes; 0 = nodata",
    }
)

# --------------------- Compute k and tau from LAI ------------------

lai_no0 = xr.where(lai_on_topo <= 0, LAI_FLOOR, lai_on_topo)

# tau (diffuse+thermal)
tau_0 = 1 - (0.29 * np.log(lai_no0) + 0.55)
tau_0 = xr.where(lai_no0 < 0.15, 1, tau_0)
tau_0 = xr.where(lai_no0 > 4.72, 0, tau_0)
tau = xr.where(vh_on_topo <= 2, 1, tau_0)

# k ("mu") for direct solar
k_0 = lai_no0 / (2 * xr.where(vh_on_topo == 0, np.nan, vh_on_topo))
k = xr.where(vh_on_topo <= 2, 0, k_0)
k = xr.where(k <= 0.074, k, 0.074).fillna(0)

k_LAI = k.astype("float32")
tau_LAI = tau.astype("float32")

# ---------------------- Apply class-based overrides ----------------

veg_type = nlcd_on_topo
veg_k = xr.zeros_like(dem, dtype="float32")
veg_tau = xr.ones_like(dem, dtype="float32")

# 1) bare/low veg -> k=0, tau=1
mask_open = np.isin(veg_type, ZERO_TAU_TYPES)
veg_k = xr.where(mask_open, 0, veg_k)
veg_tau = xr.where(mask_open, 1.0, veg_tau)

# 2) evergreen (42) -> LAI-derived
mask_42 = veg_type == 42
veg_k = xr.where(mask_42, k_LAI, veg_k)
veg_tau = xr.where(mask_42, tau_LAI, veg_tau)

# 3) deciduous (41)
mask_41 = veg_type == 41
if USE_FIXED_DECID_MIXED:
    veg_k = xr.where(mask_41, DECIDUOUS_41_k, veg_k)
    veg_tau = xr.where(mask_41, DECIDUOUS_41_tau, veg_tau)
else:
    veg_k = xr.where(mask_41, k_LAI, veg_k)
    veg_tau = xr.where(mask_41, tau_LAI, veg_tau)

# 4) mixed (43)
mask_43 = veg_type == 43
if USE_FIXED_DECID_MIXED:
    veg_k = xr.where(mask_43, MIXED_43_k, veg_k)
    veg_tau = xr.where(mask_43, MIXED_43_tau, veg_tau)
else:
    veg_k = xr.where(mask_43, k_LAI, veg_k)
    veg_tau = xr.where(mask_43, tau_LAI, veg_tau)

# ------------------------- Build projection var --------------------

projection_var = build_projection_var(tmpl, CRS_EPSG)

# -------------------------- Assemble Dataset -----------------------

ds = xr.Dataset(
    data_vars=dict(
        dem=dem,
        veg_type=veg_type,
        veg_tau=veg_tau.astype("float32"),
        veg_k=veg_k.astype("float32"),
        veg_height=vh_on_topo.astype("float32"),
        mask=mask_core_da.astype("uint8"),  # 1 = core rectangle; 0 = buffer ring & outside
        projection=projection_var,  # scalar int64 with full CF/GDAL attrs
    ),
    coords=dict(
        x=dem.x,
        y=dem.y,
    ),
    attrs=dict(
        Conventions="CF-1.6",
        dateCreated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        Title="Preparing topographic and vegetation input for SMRF/AWSM",
        history=f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created @ {TARGET_RES_M} m",
        institution="Boise State University, Department of Geosciences",
    ),
)

# Advertise the grid mapping on geospatial variables (nice-to-have; some tools expect this)
for v in ["dem", "veg_type", "veg_tau", "veg_k", "veg_height", "mask"]:
    if v in ds:
        ds[v].attrs["grid_mapping"] = "projection"

# ------------------------------ Save -------------------------------

if SAVE_OUTPUT:
    OUT_NC_PATH = Path(OUT_NC_PATH)
    OUT_NC_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {} for v in ds.data_vars}
    ds.to_netcdf(OUT_NC_PATH, encoding=encoding)
    print(f"Wrote: {OUT_NC_PATH}")
