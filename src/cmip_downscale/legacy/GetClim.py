import datetime
import operator

import numpy as np

from cmip_downscale.legacy.BioClim import BioClim
from cmip_downscale.legacy.extract import get_chelsa, get_cmip, get_esgf


class cmip6_clim:
    """
    Climatology class for monthly climatologies for CMIP6 data

    :param activity_id: the activity_id according to CMIP6
    :param table_id: the table id according to CMIP6
    :param experiment_id: the experiment_id according to CMIP6
    :param variable_id: the variable shortname according to CMIP6
    :param institution_id: the instituion_id according to CMIP6
    :param source_id: the source_id according to CMIP6
    :param member_id: the member_id according to CMIP6
    :param ref_startdate: Starting date of the reference_period
    :param ref_enddate: End date of the reference_period
    :param fut_startdate: Start date of the future future_period
    :param fut_enddate: End date of the future_period
    :param use_esgf: Use ESGF node instead of Pangeo
    :param node: string, address of the ESFG node, default=https://esgf.ceda.ac.uk/esg-search
    """

    def __init__(
        self,
        activity_id,
        table_id,
        variable_id,
        experiment_id,
        institution_id,
        source_id,
        member_id,
        ref_startdate,
        ref_enddate,
        fut_startdate,
        fut_enddate,
        use_esgf,
        node,
    ):
        self.activity_id = activity_id
        self.table_id = table_id
        self.variable_id = variable_id
        self.experiment_id = experiment_id
        self.institution_id = institution_id
        self.source_id = source_id
        self.member_id = member_id
        self.refps = ref_startdate
        self.refpe = ref_enddate
        self.fefps = fut_startdate
        self.fefpe = fut_enddate
        self.use_esgf = use_esgf
        self.node = node

        if self.use_esgf is True:
            self.future_period = (
                get_esgf(
                    activity_id=self.activity_id,
                    table_id=self.table_id,
                    variable_id=self.variable_id,
                    experiment_id=self.experiment_id,
                    source_id=self.source_id,
                    member_id=self.member_id,
                    node=self.node,
                )
                .sel(time=slice(self.fefps, self.fefpe))
                .groupby("time.month")
                .mean("time")
            )
        if self.use_esgf is False:
            self.future_period = (
                get_cmip(
                    self.activity_id,
                    self.table_id,
                    self.variable_id,
                    self.experiment_id,
                    self.institution_id,
                    self.source_id,
                    self.member_id,
                )
                .sel(time=slice(self.fefps, self.fefpe))
                .groupby("time.month")
                .mean("time")
            )
        # print("future data loaded... ")
        if self.use_esgf is True:
            self.historical_period = (
                get_esgf(
                    activity_id="CMIP6",
                    table_id=self.table_id,
                    variable_id=self.variable_id,
                    experiment_id="historical",
                    source_id=self.source_id,
                    member_id=self.member_id,
                    node=self.node,
                )
                .sel(time=slice(self.refps, self.refpe))
                .groupby("time.month")
                .mean("time")
            )
        if self.use_esgf is False:
            self.historical_period = (
                get_cmip(
                    "CMIP",
                    self.table_id,
                    self.variable_id,
                    "historical",
                    self.institution_id,
                    self.source_id,
                    self.member_id,
                )
                .sel(time=slice(self.refps, self.refpe))
                .groupby("time.month")
                .mean("time")
            )
        # print("historical period set... ")
        if self.use_esgf is True:
            self.reference_period = (
                get_esgf(
                    activity_id="CMIP6",
                    table_id=self.table_id,
                    variable_id=self.variable_id,
                    experiment_id="historical",
                    source_id=self.source_id,
                    member_id=self.member_id,
                    node=self.node,
                )
                .sel(time=slice("1981-01-15", "2010-12-15"))
                .groupby("time.month")
                .mean("time")
            )
        if self.use_esgf is False:
            self.reference_period = (
                get_cmip(
                    "CMIP",
                    self.table_id,
                    self.variable_id,
                    "historical",
                    self.institution_id,
                    self.source_id,
                    self.member_id,
                )
                .sel(time=slice("1981-01-15", "2010-12-15"))
                .groupby("time.month")
                .mean("time")
            )
        # print("reference period set... done")

    def get_anomaly(self, period):
        """
        Get climatological anomaly between the reference and future future_period

        :param period:  either future (futr) or historical (hist)
        :return: anomaly
        :rtype: xarray
        """
        if period == "futr":
            if (
                self.variable_id == "tas"
                or self.variable_id == "tasmin"
                or self.variable_id == "tasmax"
            ):
                res = self.future_period - self.reference_period  # additive anomaly
            if self.variable_id == "pr":
                res = (self.future_period * 86400 + 0.01) / (
                    self.reference_period * 86400 + 0.01
                )  # multiplicative anomaly

        if period == "hist":
            if (
                self.variable_id == "tas"
                or self.variable_id == "tasmin"
                or self.variable_id == "tasmax"
            ):
                res = self.historical_period - self.reference_period  # additive anomaly
            if self.variable_id == "pr":
                res = (self.historical_period * 86400 + 0.01) / (
                    self.reference_period * 86400 + 0.01
                )  # multiplicative anomaly

        res1 = res.assign_coords(lon=(((res.lon + 180) % 360) - 180)).sortby(
            "lon"
        )  # res1 = res.assign_coords({"lon": (((res.lon) % 360) - 180)}) #bugfix
        return res1

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
    print("start downloading CMIP data:")
    cmip_data = {
        var: cmip6_clim(
            activity_id=activity_id,
            table_id=table_id,
            variable_id=var,
            experiment_id=experiment_id,
            institution_id=institution_id,
            source_id=source_id,
            member_id=member_id,
            ref_startdate=refps,
            ref_enddate=refpe,
            fut_startdate=fefps,
            fut_enddate=fefpe,
            use_esgf=use_esgf,
            node=node,
        ) for var in ["pr", "tas", "tasmax", "tasmin"]
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

        for start, end, period in [(refps, refpe, 'hist'), (fefps, fefpe, 'futr')]:
            anomaly = cmip.get_anomaly(period)
            interp = anomaly.interp(lat=chelsa["lat"], lon=chelsa["lon"])
            result = op(chelsa, interp)

            year = get_average_year(start, end)
            result["month"] = [
                datetime.datetime(year, month, 15)
                for month in result["month"].values
            ]

            dc[period + "_" + var] = result

    print("start building climatologies data:")
    biohist = BioClim(dc["hist_pr"], dc["hist_tas"], dc["hist_tasmax"], dc["hist_tasmin"])
    biofutr = BioClim(dc["futr_pr"], dc["futr_tas"], dc["futr_tasmax"], dc["futr_tasmin"])

    if output is not None:
        file_template = f"{output}/CHELSA_{institution_id}_{source_id}_{{var}}_{experiment_id}_{member_id}_{{start}}_{{end}}.nc"

        print("saving climatologies:")
        for start, end, period in [(refps, refpe, 'hist'), (fefps, fefpe, 'futr')]:
            for var in ["pr", "tas", "tasmax", "tasmin"]:
                file_name = file_template.format(var=var, start=start, end=end)
                getattr(dc, period + "_" + var).to_netcdf(file_name)

        print("saving bioclims:")
        for start, end, data in [(refps, refpe, biohist), (fefps, fefpe, biofutr)]:
            for var in ['gdd'] + [f'bio{i}' for i in range(1, 20)]:
                file_name = file_template.format(var=var, start=start, end=end)
                getattr(data, var)().to_netcdf(file_name)


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

