"""Shared Chronos source configuration for Fetch and Load.

@scenario: Define the source sequence fields once without coupling pipeline stages.

The 44-item representative list is the default. Only an explicit
``--source chronos --subsets all`` selects all 67 supported configs, including
large WeatherBench hourly data and synthetic or augmented training corpora.
``--source all`` preserves the representative default.
"""

CHRONOS_HF_DATASET = "autogluon/chronos_datasets"

CHRONOS_REPRESENTATIVE_SUBSETS = [
    "dominick", "electricity_15min", "ercot", "exchange_rate",
    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
    "m4_weekly", "m4_yearly", "m5", "mexico_city_bikes",
    "monash_australian_electricity", "monash_car_parts", "monash_cif_2016",
    "monash_covid_deaths", "monash_electricity_hourly", "monash_electricity_weekly",
    "monash_fred_md", "monash_hospital", "monash_kdd_cup_2018",
    "monash_london_smart_meters", "monash_pedestrian_counts", "monash_rideshare",
    "monash_saugeenday", "monash_temperature_rain", "monash_tourism_monthly",
    "monash_tourism_quarterly", "monash_tourism_yearly", "monash_traffic",
    "monash_weather", "nn5", "solar", "solar_1h",
    "taxi_1h", "taxi_30min", "uber_tlc_daily", "uber_tlc_hourly",
    "ushcn_daily", "weatherbench_daily", "weatherbench_weekly",
    "wiki_daily_100k", "wind_farms_daily", "wind_farms_hourly",
]

# This list is the explicit set of Chronos configurations whose target schema
# is supported by TS3.  The representative list remains the bounded default;
# callers must request ``--subsets all`` to select this full list.
CHRONOS_SUPPORTED_SUBSETS = [
    *CHRONOS_REPRESENTATIVE_SUBSETS,
    "monash_m1_monthly", "monash_m1_quarterly", "monash_m1_yearly",
    "monash_m3_monthly", "monash_m3_quarterly", "monash_m3_yearly",
    "monash_nn5_weekly", "training_corpus_kernel_synth_1m",
    "training_corpus_tsmixup_10m",
    "weatherbench_hourly_10m_u_component_of_wind",
    "weatherbench_hourly_10m_v_component_of_wind",
    "weatherbench_hourly_2m_temperature", "weatherbench_hourly_geopotential",
    "weatherbench_hourly_potential_vorticity",
    "weatherbench_hourly_relative_humidity",
    "weatherbench_hourly_specific_humidity", "weatherbench_hourly_temperature",
    "weatherbench_hourly_toa_incident_solar_radiation",
    "weatherbench_hourly_total_cloud_cover",
    "weatherbench_hourly_total_precipitation",
    "weatherbench_hourly_u_component_of_wind",
    "weatherbench_hourly_v_component_of_wind", "weatherbench_hourly_vorticity",
]

CHRONOS_CONFIG_PATHS: dict[str, str] = {
    subset: subset for subset in CHRONOS_REPRESENTATIVE_SUBSETS
}
CHRONOS_CONFIG_PATHS.update({
    "monash_m1_monthly": "monash_m1_monthly",
    "monash_m1_quarterly": "monash_m1_quarterly",
    "monash_m1_yearly": "monash_m1_yearly",
    "monash_m3_monthly": "monash_m3_monthly",
    "monash_m3_quarterly": "monash_m3_quarterly",
    "monash_m3_yearly": "monash_m3_yearly",
    "monash_nn5_weekly": "monash_nn5_weekly",
    "training_corpus_kernel_synth_1m": "training_corpus/kernel_synth_1m",
    "training_corpus_tsmixup_10m": "training_corpus/tsmixup_10m",
    "weatherbench_hourly_10m_u_component_of_wind": "weatherbench_hourly/10m_u_component_of_wind",
    "weatherbench_hourly_10m_v_component_of_wind": "weatherbench_hourly/10m_v_component_of_wind",
    "weatherbench_hourly_2m_temperature": "weatherbench_hourly/2m_temperature",
    "weatherbench_hourly_geopotential": "weatherbench_hourly/geopotential",
    "weatherbench_hourly_potential_vorticity": "weatherbench_hourly/potential_vorticity",
    "weatherbench_hourly_relative_humidity": "weatherbench_hourly/relative_humidity",
    "weatherbench_hourly_specific_humidity": "weatherbench_hourly/specific_humidity",
    "weatherbench_hourly_temperature": "weatherbench_hourly/temperature",
    "weatherbench_hourly_toa_incident_solar_radiation": "weatherbench_hourly/toa_incident_solar_radiation",
    "weatherbench_hourly_total_cloud_cover": "weatherbench_hourly/total_cloud_cover",
    "weatherbench_hourly_total_precipitation": "weatherbench_hourly/total_precipitation",
    "weatherbench_hourly_u_component_of_wind": "weatherbench_hourly/u_component_of_wind",
    "weatherbench_hourly_v_component_of_wind": "weatherbench_hourly/v_component_of_wind",
    "weatherbench_hourly_vorticity": "weatherbench_hourly/vorticity",
})

CHRONOS_SEQUENCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "dominick": ("target",), "electricity_15min": ("consumption_kW",),
    "ercot": ("target",), "exchange_rate": ("target",), "m4_daily": ("target",),
    "m4_hourly": ("target",), "m4_monthly": ("target",), "m4_quarterly": ("target",),
    "m4_weekly": ("target",), "m4_yearly": ("target",), "m5": ("target",),
    "mexico_city_bikes": ("target",), "monash_australian_electricity": ("target",),
    "monash_car_parts": ("target",), "monash_cif_2016": ("target",),
    "monash_covid_deaths": ("target",), "monash_electricity_hourly": ("target",),
    "monash_electricity_weekly": ("target",), "monash_fred_md": ("target",),
    "monash_hospital": ("target",), "monash_kdd_cup_2018": ("target",),
    "monash_london_smart_meters": ("target",), "monash_pedestrian_counts": ("target",),
    "monash_rideshare": (
        "price_min", "price_mean", "price_max", "distance_min", "distance_mean",
        "distance_max", "surge_min", "surge_mean", "surge_max", "api_calls",
        "temp", "rain", "humidity", "clouds", "wind",
    ),
    "monash_saugeenday": ("T1",),
    "monash_temperature_rain": (
        "t_mean", "prcp_sum", "t_max", "t_min",
        "fcst_0_dailypop", "fcst_0_dailypop1", "fcst_0_dailypop10", "fcst_0_dailypop15",
        "fcst_0_dailypop25", "fcst_0_dailypop5", "fcst_0_dailypop50", "fcst_0_dailyprecip",
        "fcst_0_dailyprecip10pct", "fcst_0_dailyprecip25pct", "fcst_0_dailyprecip50pct", "fcst_0_dailyprecip75pct",
        "fcst_1_dailypop", "fcst_1_dailypop1", "fcst_1_dailypop10", "fcst_1_dailypop15",
        "fcst_1_dailypop25", "fcst_1_dailypop5", "fcst_1_dailypop50", "fcst_1_dailyprecip",
        "fcst_1_dailyprecip10pct", "fcst_1_dailyprecip25pct", "fcst_1_dailyprecip50pct", "fcst_1_dailyprecip75pct",
        "fcst_2_dailypop", "fcst_2_dailypop1", "fcst_2_dailypop10", "fcst_2_dailypop15",
        "fcst_2_dailypop25", "fcst_2_dailypop5", "fcst_2_dailypop50", "fcst_2_dailyprecip",
        "fcst_2_dailyprecip10pct", "fcst_2_dailyprecip25pct", "fcst_2_dailyprecip50pct", "fcst_2_dailyprecip75pct",
        "fcst_3_dailypop", "fcst_3_dailypop1", "fcst_3_dailypop10", "fcst_3_dailypop15",
        "fcst_3_dailypop25", "fcst_3_dailypop5", "fcst_3_dailypop50", "fcst_3_dailyprecip",
        "fcst_3_dailyprecip10pct", "fcst_3_dailyprecip25pct", "fcst_3_dailyprecip50pct", "fcst_3_dailyprecip75pct",
        "fcst_4_dailypop", "fcst_4_dailypop1", "fcst_4_dailypop10", "fcst_4_dailypop15",
        "fcst_4_dailypop25", "fcst_4_dailypop5", "fcst_4_dailypop50", "fcst_4_dailyprecip",
        "fcst_4_dailyprecip10pct", "fcst_4_dailyprecip25pct", "fcst_4_dailyprecip50pct", "fcst_4_dailyprecip75pct",
        "fcst_5_dailypop", "fcst_5_dailypop1", "fcst_5_dailypop10", "fcst_5_dailypop15",
        "fcst_5_dailypop25", "fcst_5_dailypop5", "fcst_5_dailypop50", "fcst_5_dailyprecip",
        "fcst_5_dailyprecip10pct", "fcst_5_dailyprecip25pct", "fcst_5_dailyprecip50pct", "fcst_5_dailyprecip75pct",
    ),
    "monash_tourism_monthly": ("target",), "monash_tourism_quarterly": ("target",),
    "monash_tourism_yearly": ("target",), "monash_traffic": ("target",),
    "monash_weather": ("target",), "nn5": ("target",), "solar": ("power_mw",),
    "solar_1h": ("power_mw",), "taxi_1h": ("target",), "taxi_30min": ("target",),
    "uber_tlc_daily": ("target",), "uber_tlc_hourly": ("target",),
    "ushcn_daily": ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN"),
    "weatherbench_daily": ("target",), "weatherbench_weekly": ("target",),
    "wiki_daily_100k": ("target",), "wind_farms_daily": ("target",),
    "wind_farms_hourly": ("target",),
    "monash_m1_monthly": ("target",), "monash_m1_quarterly": ("target",),
    "monash_m1_yearly": ("target",), "monash_m3_monthly": ("target",),
    "monash_m3_quarterly": ("target",), "monash_m3_yearly": ("target",),
    "monash_nn5_weekly": ("target",),
    "training_corpus_kernel_synth_1m": ("target",),
    "training_corpus_tsmixup_10m": ("target",),
    "weatherbench_hourly_10m_u_component_of_wind": ("target",),
    "weatherbench_hourly_10m_v_component_of_wind": ("target",),
    "weatherbench_hourly_2m_temperature": ("target",),
    "weatherbench_hourly_geopotential": ("target",),
    "weatherbench_hourly_potential_vorticity": ("target",),
    "weatherbench_hourly_relative_humidity": ("target",),
    "weatherbench_hourly_specific_humidity": ("target",),
    "weatherbench_hourly_temperature": ("target",),
    "weatherbench_hourly_toa_incident_solar_radiation": ("target",),
    "weatherbench_hourly_total_cloud_cover": ("target",),
    "weatherbench_hourly_total_precipitation": ("target",),
    "weatherbench_hourly_u_component_of_wind": ("target",),
    "weatherbench_hourly_v_component_of_wind": ("target",),
    "weatherbench_hourly_vorticity": ("target",),
}
