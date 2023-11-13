import datetime
import operator

from cmip_downscale.legacy.BioClim import BioClim
from cmip_downscale.legacy.extract import get_chelsa, get_cmip, get_esgf
from cmip_downscale.legacy.transform import (
    additive_anomaly,
    multiplicative_anomaly,
    subset_ma,
    get_average_year,
)


def chelsa_cmip6(
    source_id: str,
    institution_id: str,
    table_id: str,
    activity_id: str,
    experiment_id: str,
    member_id: str,
    refps: str,
    refpe: str,
    fefps: str,
    fefpe: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    output: str = None,
    use_esgf: bool = False,
    node: str = "https://esgf.ceda.ac.uk/esg-search",
):
    """Calculate chelsa cmip 6 climatological normals and bio-climatic variables

    Parameters
    ----------
    source_id
        Source model (GCM), e.g. MPI-ESM1-2-LR.
    institution_id
        Institution ID, e.g. MPI-M.
    table_id
        Table ID, e.g. Amon.
    activity_id
        Activity ID, e.g. CMIP.
    experiment_id
        Experiment ID, e.g. historical.
    member_id
        Member ID, e.g. r1i1p1f1.
    refps
        Starting date of the reference_period, e.g. 1981-01-01.
    refpe
        End date of the reference_period, e.g. 2010-12-31.
    fefps
        Start date of the future future_period, e.g. 2071-01-01.
    fefpe
        End date of the future_period, e.g. 2100-12-31.
    xmin
        Minimum longitude [Decimal degree].
    xmax
        Maximum longitude [Decimal degree].
    ymin
        Minimum latitude [Decimal degree].
    ymax
        Maximum latitude [Decimal degree].
    output
        Directory to write results to.
    use_esgf
        Use ESGF node instead of Pangeo, default=False.
    node
        Address of the ESFG node, default=https://esgf.ceda.ac.uk/esg-search.
    """
    kwargs = {
        "table_id": table_id,
        "source_id": source_id,
        "member_id": member_id,
    }
    if use_esgf:
        loader = get_esgf
        kwargs['node'] = node
        historical_activity_id = "CMIP6"
    else:
        loader = get_cmip
        kwargs['institution_id'] = institution_id
        historical_activity_id = "CMIP"

    print("start downloading CMIP data:")
    cmip_data = {}
    for var in ["pr", "tas", "tasmax", "tasmin"]:
        reference_period = loader(
            variable_id=var,
            activity_id=historical_activity_id,
            experiment_id="historical",
            **kwargs,
        )
        future_period = loader(
            variable_id=var,
            activity_id=activity_id,
            experiment_id=experiment_id,
            **kwargs,
        )
        cmip_data[var] = {
            "reference": subset_ma(reference_period, refps, refpe),
            "future": subset_ma(future_period, fefps, fefpe),
        }

    print(
        "start downloading CHELSA data (depending on your internet speed this might take a while...)"
    )
    chelsa_data = {
        var: get_chelsa(var, xmin, xmax, ymin, ymax)
        for var in ["pr", "tas", "tasmax", "tasmin"]
    }

    print("applying delta change:")
    dc = {}
    for var in ["pr", "tas", "tasmax", "tasmin"]:
        cmip = cmip_data[var]
        chelsa = chelsa_data[var]
        op = operator.mul if var in ["pr"] else operator.add
        get_anomaly = multiplicative_anomaly if var in ["pr"] else additive_anomaly

        anomaly = get_anomaly(cmip['reference'], cmip['future'])
        interp = anomaly.interp(lat=chelsa["lat"], lon=chelsa["lon"])
        result = op(chelsa, interp)

        year = get_average_year(fefps, fefpe)
        result["month"] = [
            datetime.datetime(year, month, 15)
            for month in result["month"].values
        ]

        dc[var] = result

    print("start building climatologies data:")
    bioclim = BioClim(dc["pr"], dc["tas"], dc["tasmax"], dc["tasmin"])

    if output is not None:
        file_template = f"{output}/CHELSA_{institution_id}_{source_id}_{{var}}_{experiment_id}_{member_id}_{fefps}_{fefpe}.nc"

        print("saving climatologies:")
        for var in ["pr", "tas", "tasmax", "tasmin"]:
            file_name = file_template.format(var=var)
            dc[var].to_netcdf(file_name)

        print("saving bioclims:")
        for var in ['gdd'] + [f'bio{i}' for i in range(1, 20)]:
            file_name = file_template.format(var=var)
            getattr(bioclim, var)().to_netcdf(file_name)
