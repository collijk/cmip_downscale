import fsspec
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from pyesgf.search import SearchConnection


def get_esgf(
    activity_id: str,
    table_id: str,
    variable_id: str,
    experiment_id: str,
    source_id: str,
    member_id: str,
    frequency_id="mon",
    node="https://esgf.ceda.ac.uk/esg-search",
) -> xr.Dataset:
    """Get CMIP model from ESGF via lazy loading.

    Parameters
    ----------
    activity_id
        Activity ID, e.g. CMIP.
    table_id
        Table ID, e.g. Amon.
    variable_id
        Variable ID, e.g. tas.
    experiment_id
        Experiment ID, e.g. historical.
    source_id
        Source ID, e.g. HadGEM3-GC31-LL.
    member_id
        Member ID, e.g. r1i1p1f3.
    frequency_id
        Frequency ID, e.g. mon.
    node
        ESGF node to use.

    Returns
    -------
    xarray.Dataset
        The results for the particular CMIP model.
    """
    conn = SearchConnection(node, distrib=True)

    ctx = conn.new_context(
        project=activity_id,
        source_id=source_id,
        table_id=table_id,
        experiment_id=experiment_id,
        variable=variable_id,
        variant_label=member_id,
        frequency=frequency_id,
    )

    result = ctx.search()[0]
    files = result.file_context().search()

    ff = []
    for file in files:
        print(file.opendap_url)
        ff.append(file.opendap_url)

    ds = xr.open_mfdataset(ff, combine="nested", concat_dim="time")

    ds["time"] = np.sort(ds["time"].values)

    return ds


def get_cmip(
    activity_id: str,
    table_id: str,
    variable_id: str,
    experiment_id: str,
    institution_id: str,
    source_id: str,
    member_id: str,
) -> xr.Dataset:
    """Get CMIP model from Google Cloud Storage via lazy loading.

    Parameters
    ----------
    activity_id
        Activity ID, e.g. CMIP.
    table_id
        Table ID, e.g. Amon.
    variable_id
        Variable ID, e.g. tas.
    experiment_id
        Experiment ID, e.g. historical.
    institution_id
        Institution ID, e.g. MOHC.
    source_id
        Source ID, e.g. HadGEM3-GC31-LL.
    member_id
        Member ID, e.g. r1i1p1f3.

    Returns
    -------
    xarray.Dataset
        The results for the particular CMIP model.
    """

    gcs = gcsfs.GCSFileSystem(token="anon")
    df = pd.read_csv(
        "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
    )
    search_string = (
        "activity_id == '"
        + activity_id
        + "' & table_id == '"
        + table_id
        + "' & variable_id == '"
        + variable_id
        + "' & experiment_id == '"
        + experiment_id
        + "' & institution_id == '"
        + institution_id
        + "' & source_id == '"
        + source_id
        + "' & member_id == '"
        + member_id
        + "'"
    )
    df_ta = df.query(search_string)
    # get the path to a specific zarr store (the first one from the dataframe above)
    zstore = df_ta.zstore.values[-1]
    # create a mutable-mapping-style interface to the store
    mapper = gcs.get_mapper(zstore)
    # open it using xarray and zarr
    ds = xr.open_zarr(mapper, consolidated=True)
    ds["time"] = np.sort(ds["time"].values)

    return ds


def get_chelsa(
    variable_id: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> xr.Dataset:
    """Download data from chelsa"""
    chelsa_root = "https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies/1981-2010/ncdf/CHELSA"

    data = []
    for month in range(1, 13):
        url = f"{chelsa_root}_{variable_id}_{month:02d}_1981-2010_V.2.1.nc"

        with fsspec.open(url) as fobj:
            ds = xr.open_dataset(fobj).chunk({"lat": 500, "lon": 500})
            mask_lon = (ds.lon >= xmin) & (ds.lon <= xmax)
            mask_lat = (ds.lat >= ymin) & (ds.lat <= ymax)
            ds = ds.where(mask_lon & mask_lat, drop=True)
            ds.load()
        data.append(ds)

    ds = xr.concat(data, "time")
    res = ds.assign(Band1=ds["Band1"] * 0.1)

    return res
