"""Frozen cross-source Fetch Profile selections.

This module contains only the versioned Profile input. It does not inspect
remote repositories, read environment variables, or import source defaults.
"""

PROFILE_SCHEMA_VERSION = 1

PROFILE_SPECS = {
    "1b": {
        "estimate_method": "public_metadata_v0_without_blocked_sources",
        "estimated_observations": 860_000_000,
        "sample_probability": 0.11,
        "sources": (
            {
                "source": "chronos",
                "revision": "main",
                "subsets": (
                    "dominick", "electricity_15min", "ercot", "exchange_rate",
                    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
                    "m4_weekly", "m4_yearly", "m5", "mexico_city_bikes",
                    "monash_australian_electricity", "monash_car_parts", "monash_cif_2016",
                    "monash_covid_deaths", "monash_electricity_hourly", "monash_electricity_weekly",
                    "monash_fred_md", "monash_hospital", "monash_kdd_cup_2018",
                    "monash_london_smart_meters", "monash_pedestrian_counts", "monash_rideshare",
                    "monash_saugeenday", "monash_temperature_rain", "monash_tourism_monthly",
                    "monash_tourism_quarterly", "monash_tourism_yearly", "monash_traffic",
                    "monash_weather", "nn5", "solar", "solar_1h", "taxi_1h", "taxi_30min",
                    "uber_tlc_daily", "uber_tlc_hourly", "ushcn_daily", "weatherbench_daily",
                    "weatherbench_weekly", "wiki_daily_100k", "wind_farms_daily", "wind_farms_hourly",
                ),
            },
            {
                "source": "lotsa",
                "revision": "main",
                "subsets": (
                    "BEIJING_SUBWAY_30MIN", "HZMETRO", "PEMS03", "PEMS04", "PEMS07",
                    "PEMS08", "PEMS_BAY",
                ),
            },
            {
                "source": "monash",
                "revision": "main",
                "subsets": (
                    "australian_electricity_demand", "cif_2016", "covid_deaths", "fred_md",
                    "hospital", "kaggle_web_traffic_weekly", "nn5_weekly", "oikolab_weather",
                    "pedestrian_counts", "saugeenday", "solar_10_minutes", "solar_4_seconds",
                    "solar_weekly", "tourism_monthly", "tourism_quarterly", "tourism_yearly",
                    "traffic_hourly", "traffic_weekly", "us_births", "weather", "wind_4_seconds",
                ),
            },
            {
                "source": "quito",
                "revision": "main",
                "subsets": (
                    "v20260315/pretrain_hour-00001-of-00001.parquet",
                    "v20260315/pretrain_min-00001-of-00001.parquet",
                ),
            },
            {
                "source": "time300b",
                "revision": "main",
                "subsets": ("energy/buildings_900k",),
            },
            {
                "source": "utsd",
                "revision": "main",
                "utsd_config": "UTSD-1G",
            },
        ),
    },
    "4b": {
        "estimate_method": "public_metadata_v0_without_blocked_sources",
        "estimated_observations": 3_560_000_000,
        "sample_probability": 0.32,
        "sources": (
            {
                "source": "chronos",
                "revision": "main",
                "subsets": (
                    "dominick", "electricity_15min", "ercot", "exchange_rate",
                    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
                    "m4_weekly", "m4_yearly", "m5", "mexico_city_bikes",
                    "monash_australian_electricity", "monash_car_parts", "monash_cif_2016",
                    "monash_covid_deaths", "monash_electricity_hourly", "monash_electricity_weekly",
                    "monash_fred_md", "monash_hospital", "monash_kdd_cup_2018",
                    "monash_london_smart_meters", "monash_pedestrian_counts", "monash_rideshare",
                    "monash_saugeenday", "monash_temperature_rain", "monash_tourism_monthly",
                    "monash_tourism_quarterly", "monash_tourism_yearly", "monash_traffic",
                    "monash_weather", "nn5", "solar", "solar_1h", "taxi_1h", "taxi_30min",
                    "training_corpus_kernel_synth_1m", "uber_tlc_daily", "uber_tlc_hourly",
                    "ushcn_daily", "weatherbench_daily", "weatherbench_weekly", "wiki_daily_100k",
                    "wind_farms_daily", "wind_farms_hourly",
                ),
            },
            {
                "source": "lotsa",
                "revision": "main",
                "subsets": (
                    "BEIJING_SUBWAY_30MIN", "HZMETRO", "PEMS03", "PEMS04", "PEMS07",
                    "PEMS08", "PEMS_BAY",
                ),
            },
            {
                "source": "monash",
                "revision": "main",
                "subsets": (
                    "australian_electricity_demand", "cif_2016", "covid_deaths", "fred_md",
                    "hospital", "kaggle_web_traffic_weekly", "nn5_weekly", "oikolab_weather",
                    "pedestrian_counts", "saugeenday", "solar_10_minutes", "solar_4_seconds",
                    "solar_weekly", "tourism_monthly", "tourism_quarterly", "tourism_yearly",
                    "traffic_hourly", "traffic_weekly", "us_births", "weather", "wind_4_seconds",
                ),
            },
            {
                "source": "quito",
                "revision": "main",
                "subsets": (
                    "v20260315/pretrain_hour-00001-of-00001.parquet",
                    "v20260315/pretrain_min-00001-of-00001.parquet",
                ),
            },
            {
                "source": "time300b",
                "revision": "main",
                "subsets": (
                    "energy/buildings_900k",
                    "nature/weatherbench_hourly_geopotential",
                ),
            },
            {
                "source": "utsd",
                "revision": "main",
                "utsd_config": "UTSD-1G",
            },
        ),
    },
    "10b": {
        "estimate_method": "public_metadata_v0_without_blocked_sources",
        "estimated_observations": 9_370_000_000,
        "sample_probability": 0.40,
        "sources": (
            {
                "source": "chronos",
                "revision": "main",
                "subsets": (
                    "dominick", "electricity_15min", "ercot", "exchange_rate",
                    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
                    "m4_weekly", "m4_yearly", "m5", "mexico_city_bikes",
                    "monash_australian_electricity", "monash_car_parts", "monash_cif_2016",
                    "monash_covid_deaths", "monash_electricity_hourly", "monash_electricity_weekly",
                    "monash_fred_md", "monash_hospital", "monash_kdd_cup_2018",
                    "monash_london_smart_meters", "monash_pedestrian_counts", "monash_rideshare",
                    "monash_saugeenday", "monash_temperature_rain", "monash_tourism_monthly",
                    "monash_tourism_quarterly", "monash_tourism_yearly", "monash_traffic",
                    "monash_weather", "nn5", "solar", "solar_1h", "taxi_1h", "taxi_30min",
                    "training_corpus_kernel_synth_1m", "training_corpus_tsmixup_10m",
                    "uber_tlc_daily", "uber_tlc_hourly", "ushcn_daily", "weatherbench_daily",
                    "weatherbench_weekly", "wiki_daily_100k", "wind_farms_daily", "wind_farms_hourly",
                ),
            },
            {
                "source": "lotsa",
                "revision": "main",
                "subsets": (
                    "BEIJING_SUBWAY_30MIN", "HZMETRO", "PEMS03", "PEMS04", "PEMS07",
                    "PEMS08", "PEMS_BAY",
                ),
            },
            {
                "source": "monash",
                "revision": "main",
                "subsets": (
                    "australian_electricity_demand", "cif_2016", "covid_deaths", "fred_md",
                    "hospital", "kaggle_web_traffic_weekly", "nn5_weekly", "oikolab_weather",
                    "pedestrian_counts", "saugeenday", "solar_10_minutes", "solar_4_seconds",
                    "solar_weekly", "tourism_monthly", "tourism_quarterly", "tourism_yearly",
                    "traffic_hourly", "traffic_weekly", "us_births", "weather", "wind_4_seconds",
                ),
            },
            {
                "source": "quito",
                "revision": "main",
                "subsets": (
                    "v20260315/pretrain_hour-00001-of-00001.parquet",
                    "v20260315/pretrain_min-00001-of-00001.parquet",
                ),
            },
            {
                "source": "time300b",
                "revision": "main",
                "subsets": (
                    "energy/buildings_900k",
                    "nature/weatherbench_hourly_geopotential",
                    "nature/weatherbench_hourly_temperature",
                ),
            },
            {
                "source": "utsd",
                "revision": "main",
                "utsd_config": "UTSD-1G",
            },
        ),
    },
    "profile-10b": {
        "estimate_method": "public_metadata_v1_classic_full_subsets_utsd_4g",
        "estimated_observations": 10_100_000_000,
        "sample_probability": None,
        "sources": (
            {
                "source": "chronos",
                "revision": "main",
                "subsets": (
                    "electricity_15min", "exchange_rate", "m4_daily", "m4_hourly",
                    "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly",
                    "solar", "solar_1h", "taxi_30min", "weatherbench_daily",
                    "weatherbench_weekly",
                ),
            },
            {
                "source": "gifteval",
                "revision": "main",
                "subsets": ("PEMS03",),
            },
            {
                "source": "lotsa",
                "revision": "main",
                "subsets": ("PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS_BAY"),
            },
            {
                "source": "monash",
                "revision": "main",
                "subsets": (
                    "australian_electricity_demand", "cif_2016", "fred_md",
                    "nn5_weekly", "solar_10_minutes", "tourism_monthly",
                    "tourism_quarterly", "tourism_yearly", "traffic_hourly",
                    "traffic_weekly", "weather",
                ),
            },
            {
                "source": "pile",
                "revision": "main",
                "subsets": ("classification/UCR/ACSF1",),
            },
            {
                "source": "quito",
                "revision": "main",
                "subsets": (
                    "v20260315/pretrain_hour-00001-of-00001.parquet",
                    "v20260315/pretrain_min-00001-of-00001.parquet",
                ),
            },
            {
                "source": "rmisc",
                "revision": "smaller_version",
                "subsets": ("ACSF1",),
            },
            {
                "source": "time300b",
                "revision": "main",
                "subsets": ("energy/australian_electricity_demand",),
            },
            {
                "source": "utsd",
                "revision": "main",
                "utsd_config": "UTSD-4G",
            },
        ),
    },
}
PROFILE_SPECS["profile-middle"] = PROFILE_SPECS["profile-10b"]
