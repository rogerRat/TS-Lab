"""Shared Monash source configuration for Fetch and Load.

Default Fetch and explicit ``--subsets all`` both select only configurations
without upstream-declared missing values. The excluded configurations remain
available as explicit named selections; ``--source all`` preserves the same
quality-filtered default and does not override it.
"""

MONASH_HF_DATASET = "Monash-University/monash_tsf"

MONASH_SUBSETS = [
    "weather", "electricity", "traffic", "tourism_monthly",
    "nn5_daily", "hospital", "car_parts",
]

MONASH_HF_CONFIGS = [
    "australian_electricity_demand", "bitcoin", "car_parts", "cif_2016",
    "covid_deaths", "fred_md", "hospital", "kaggle_web_traffic",
    "kaggle_web_traffic_weekly", "kdd_cup_2018", "london_smart_meters",
    "nn5_daily", "nn5_weekly", "oikolab_weather", "pedestrian_counts",
    "rideshare", "saugeenday", "solar_10_minutes", "solar_4_seconds",
    "solar_weekly", "sunspot", "temperature_rain", "tourism_monthly",
    "tourism_quarterly", "tourism_yearly", "traffic_hourly", "traffic_weekly",
    "us_births", "vehicle_trips", "weather", "wind_4_seconds",
    "wind_farms_minutely",
]

# These upstream archives explicitly identify themselves as containing missing
# values. They remain valid explicit selections but are excluded from every
# default or ``all`` Fetch selection.
MONASH_MISSING_VALUE_CONFIGS = [
    "bitcoin",
    "car_parts",
    "kaggle_web_traffic",
    "kdd_cup_2018",
    "london_smart_meters",
    "nn5_daily",
    "rideshare",
    "sunspot",
    "temperature_rain",
    "vehicle_trips",
    "wind_farms_minutely",
]

MONASH_DEFAULT_CONFIGS = [
    config
    for config in MONASH_HF_CONFIGS
    if config not in MONASH_MISSING_VALUE_CONFIGS
]

MONASH_HF_ZIP_MAP = {
    "australian_electricity_demand": "australian_electricity_demand_dataset",
    "bitcoin": "bitcoin_dataset_with_missing_values",
    "car_parts": "car_parts_dataset_with_missing_values",
    "cif_2016": "cif_2016_dataset",
    "covid_deaths": "covid_deaths_dataset",
    "fred_md": "fred_md_dataset",
    "hospital": "hospital_dataset",
    "kaggle_web_traffic": "kaggle_web_traffic_dataset_with_missing_values",
    "kaggle_web_traffic_weekly": "kaggle_web_traffic_weekly_dataset",
    "kdd_cup_2018": "kdd_cup_2018_dataset_with_missing_values",
    "london_smart_meters": "london_smart_meters_dataset_with_missing_values",
    "nn5_daily": "nn5_daily_dataset_with_missing_values",
    "nn5_weekly": "nn5_weekly_dataset",
    "oikolab_weather": "oikolab_weather_dataset",
    "pedestrian_counts": "pedestrian_counts_dataset",
    "rideshare": "rideshare_dataset_with_missing_values",
    "saugeenday": "saugeenday_dataset",
    "solar_10_minutes": "solar_10_minutes_dataset",
    "solar_4_seconds": "solar_4_seconds_dataset",
    "solar_weekly": "solar_weekly_dataset",
    "sunspot": "sunspot_dataset_with_missing_values",
    "temperature_rain": "temperature_rain_dataset_with_missing_values",
    "tourism_monthly": "tourism_monthly_dataset",
    "tourism_quarterly": "tourism_quarterly_dataset",
    "tourism_yearly": "tourism_yearly_dataset",
    "traffic_hourly": "traffic_hourly_dataset",
    "traffic_weekly": "traffic_weekly_dataset",
    "us_births": "us_births_dataset",
    "vehicle_trips": "vehicle_trips_dataset_with_missing_values",
    "weather": "weather_dataset",
    "wind_4_seconds": "wind_4_seconds_dataset",
    "wind_farms_minutely": "wind_farms_minutely_dataset_with_missing_values",
}

MONASH_NAME_MAP = {
    "weather": "weather",
    "electricity": "australian_electricity_demand",
    "traffic": "traffic_hourly",
    "tourism_monthly": "tourism_monthly",
    "nn5_daily": "nn5_daily",
    "hospital": "hospital",
    "car_parts": "car_parts",
}
