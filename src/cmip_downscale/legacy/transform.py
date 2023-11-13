import datetime

import numpy as np
import xarray as xr


def additive_anomaly(reference: xr.Dataset, future: xr.Dataset) -> xr.Dataset:
    """Compute the additive anomaly between two datasets.

    Parameters
    ----------
    reference
        Reference dataset.
    future
        Future dataset.

    Returns
    -------
    xarray.Dataset
        The additive anomaly between the two datasets.
    """
    return correct_lon(future - reference)


def multiplicative_anomaly(reference: xr.Dataset, future: xr.Dataset) -> xr.Dataset:
    """Compute the multiplicative anomaly between two datasets.

    Parameters
    ----------
    reference
        Reference dataset.
    future
        Future dataset.

    Returns
    -------
    xarray.Dataset
        The multiplicative anomaly between the two datasets.
    """
    res = ((future * 86400 + 0.01) / (reference * 86400 + 0.01))

    res = (res
           .assign_coords(lon=(((res.lon + 180) % 360) - 180))
           .sortby("lon"))
    return res


def correct_lon(ds: xr.Dataset) -> xr.Dataset:
    """Correct the longitude coordinates of a dataset.

    Parameters
    ----------
    ds
        Dataset to correct.

    Returns
    -------
    xarray.Dataset
        The corrected dataset.
    """
    res = (ds
           .assign_coords(lon=(((ds.lon + 180) % 360) - 180))
           .sortby("lon"))
    return res


def subset_ma(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    return ds.sel(time=slice(start, end)).groupby("time.month").mean("time")


def get_average_year(date_start_string: str, date_end_string: str) -> int:
    """Get the average year between two dates.

    Parameters
    ----------
    date_start_string
        Start date in string format, e.g. 1981-01-01.
    date_end_string
        End date in string format, e.g. 2010-12-31.

    Returns
    -------
    int
        The average year between the two dates.
    """
    date_start = datetime.datetime.strptime(date_start_string, "%Y-%m-%d")
    date_end = datetime.datetime.strptime(date_end_string, "%Y-%m-%d")
    return np.mean([date_start.year, date_end.year]).__round__()
