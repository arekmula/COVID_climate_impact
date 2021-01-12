import pandas as pd
import netCDF4
import numpy as np

from pathlib import Path

from matplotlib import pyplot as plt
from netCDF4 import Dataset


def read_covid_time_series_data(path: str):
    """
    Read covid time series data to pandas dataframe

    :param path: dataframe to csv file
    :return: pandas dataframe where each row represents country/province and columns are:
    1 - Country and region
    2 - Latitude
    3 - Longitude
    >=4 - Cases per day
    """
    try:
        dataframe = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File {path} doesn't exist!")
        return None

    dataframe.insert(0, "Country_region", 0)
    # Convert Nans to empty strings
    dataframe["Province/State"].fillna('', inplace=True)
    # Create one common column with Country and region after that
    dataframe["Country_region"] = dataframe["Country/Region"] + "_" + dataframe["Province/State"]
    dataframe["Country_region"] = dataframe["Country_region"].replace("_$", "", regex=True)
    # Delete previous used columns
    dataframe.pop("Country/Region")
    dataframe.pop("Province/State")
    # Set country region as index
    dataframe.set_index("Country_region", inplace=True)
    # Change columns to datetime format
    dataframe.columns.values[2:] = pd.to_datetime(dataframe.columns.values[2:])

    return dataframe


def clear_countries_with_no_death_data(df_confirmed: pd.DataFrame,
                                       df_deaths: pd.DataFrame,
                                       df_recovered: pd.DataFrame):
    """
    Delete countries that doesn't publish death data to match countries in all dataset.

    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: cleaned dataframes
    """

    # Delete rows which has only zeros
    df_deaths = df_deaths.loc[(df_deaths != 0).any(axis=1)]

    countries_with_death_info = df_deaths.index.values
    df_recovered = df_recovered[df_recovered.index.isin(countries_with_death_info)]
    df_confirmed = df_confirmed[df_confirmed.index.isin(countries_with_death_info)]

    # Some of the countries appear in death and confirmed data but don't appear in recovery data. Delete them
    countries_with_recovered_info = df_recovered.index.values
    df_deaths = df_deaths[df_deaths.index.isin(countries_with_recovered_info)]
    df_confirmed = df_confirmed[df_confirmed.index.isin(countries_with_recovered_info)]

    return df_confirmed, df_deaths, df_recovered


def calculate_recovery_data(df_recovered, df_confirmed, days_to_heal=14):
    """
    Calculates recovery data for countries that don't publish that info. It's done by moving confirmed cases to
     recovered cases with delay equal to days_to_heal

    :param days_to_heal: number of days that takes person to heal.
    :param df_recovered: dataframe with recovered cases
    :param df_confirmed: dataframe with confirmed cases
    :return:
    """
    mask = (df_recovered != 0).any(axis=1)
    countries_with_no_recovery_data = mask[mask == 0].index
    countries_indexes = []
    for country in countries_with_no_recovery_data:
        countries_indexes.append(df_recovered.index.get_loc(country))

    new_df_recovered = df_recovered.copy()
    new_df_recovered.iloc[countries_indexes, days_to_heal:] = df_confirmed.iloc[countries_indexes, :-days_to_heal]

    return new_df_recovered


def calculate_active_cases_per_day(df_confirmed: pd.DataFrame, df_deaths: pd.DataFrame, df_recovered: pd.DataFrame):
    """
    Calculates active cases per day

    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: dataframe with active cases per day
    """

    df_active_cases = (df_confirmed - df_deaths - df_recovered)

    return df_active_cases


def create_long_lat_dataframe(df_confirmed: pd.DataFrame, df_deaths: pd.DataFrame, df_recovered: pd.DataFrame):
    """
    Pop longitude and latitude from dataframes for easier usage

    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: dataframes without Lat and Long column and dataframe with Lat and Long column
    """
    df_lat_long = df_confirmed[["Lat", "Long"]]

    for df in [df_confirmed, df_deaths, df_recovered]:
        df.pop("Lat")
        df.pop("Long")

    return df_confirmed, df_deaths, df_recovered, df_lat_long


def plot_country(country_name, dataframe: pd.DataFrame, plot_title=""):
    plt.figure()
    row = dataframe.loc[country_name, :]
    row.plot(title=plot_title)


def calculate_monthly_death_ratio(df_confirmed: pd.DataFrame, df_deaths: pd.DataFrame, df_recovered: pd.DataFrame):
    """
    Calculates accumulated monthly death ratio by calculating ratio of deaths to recoveries per month

    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: dataframe with death ratio
    """
    # Delete January 2020 and January 2021 as they are not full months.
    # My dataset starts from 2020-01-22 and ends on 2021-01-06
    month_end_list = pd.date_range(df_confirmed.columns[0], df_confirmed.columns[-1], freq="M")[1:]  # Delete
    month_start_list = pd.date_range(df_confirmed.columns[0], df_confirmed.columns[-1], freq="MS")[:-1]

    df_death_ratio = pd.DataFrame()
    for month_start, month_end in zip(month_start_list, month_end_list):
        df_death_ratio[month_start] = ((df_deaths[month_end] - df_deaths[month_start]) /
                                       (df_recovered[month_end] - df_recovered[month_start]))

    df_death_ratio.fillna(0, inplace=True)
    # TODO: What with countries that stopped publishing deaths and recovered? for example China_Hubei
    # TODO: What with countries that started publishing recovery data later (for example Poland)
    # plot_country("Netherlands", df_death_ratio, "death_ratio")
    # plot_country("Netherlands", df_deaths, "deaths")
    # plot_country("Netherlands", df_recovered, "recovered")
    # plot_country("Netherlands", df_confirmed, "confirmed")
    #
    # plt.show()

    return df_death_ratio


def calculate_mean_active_cases(df_active_cases: pd.DataFrame, mean_days_number=7):
    """
    Calculate mean of active cases from last mean_days_number

    :param df_active_cases: dataframe with active cases
    :param mean_days_number: number of days to calculate mean
    :return: dataframe with mean active cases
    """

    df_mean_active_cases = pd.DataFrame(columns=df_active_cases.columns.values[mean_days_number:],
                                        index=df_active_cases.index)

    df_active_cases_copy = df_active_cases.copy()
    # If active case in day is less than 100, then put NAN in that place. This will ensure, that we will skip the data
    # that has less than 100 cases per day in calculating mean active cases in the "mean_days_number" period.
    df_active_cases_copy[df_active_cases_copy < 100] = np.nan

    for day in df_mean_active_cases.columns:
        date_range = pd.date_range(day - pd.DateOffset(days=mean_days_number - 1), day, freq="D")
        df_mean_active_cases[day] = df_active_cases_copy[date_range].mean(axis=1, skipna=True)

    return df_mean_active_cases


def calculate_reproduction_coeff(df_active_cases, reproduction_days=5):
    """
    Calculate reproduction coefficient for each day
    R_i = M_i/M_(i-rd)

    M_i - mean active cases for day i
    rd - reproduction days

    :param df_active_cases: dataframe with active cases
    :param reproduction_days: reproduction days
    :return: dataframe with reproduction coefficient
    """

    df_mean_active_cases = calculate_mean_active_cases(df_active_cases, mean_days_number=7)

    df_reproduction = pd.DataFrame(columns=df_mean_active_cases.columns.values[reproduction_days:],
                                   index=df_mean_active_cases.index)

    for day in df_reproduction:
        reproduction_day = day - pd.DateOffset(days=reproduction_days)
        df_reproduction[day] = df_mean_active_cases[day] / df_mean_active_cases[reproduction_day]

    # Drop countries for which we couldn't count any reproduction coefficient. (No more active cases than
    # 100 during whole data range)
    df_reproduction.dropna(axis=0, how='all', inplace=True)
    return df_reproduction


def find_long_lat_index(latitudes, longitudes, df_lat_long: pd.DataFrame):
    """
    Finds longitude and latitude indexes in latitudes & longitudes arrays for each country  
    
    :param latitudes: array with latitudes
    :param longitudes: array with longitudes
    :param df_lat_long: dataframe with longitude and 
    :return df_lat_long_indexes: dataframe with latitude and longitude indexes in input arrays for each country
    """

    df_lat_long_indexes = df_lat_long.copy()

    for index, country in df_lat_long.iterrows():
        country_lat = country["Lat"]
        country_long = country["Long"]
        df_lat_long_indexes.loc[index, "lat_idx"] = (np.abs(latitudes - country_lat).argmin())
        df_lat_long_indexes.loc[index, "long_idx"] = (np.abs(longitudes - country_long).argmin())

    return df_lat_long_indexes


def read_terraclimate(path_temp_max: Path, path_temp_min: Path, df_lat_long: pd.DataFrame,
                      df_temp_mean_path=Path("data/df_temp_mean.csv")):
    """
    Computes mean temperature for each country and month, based on country latitude and longitude.

    :param path_temp_max: path to file with maximum temperature for each month and lat&long
    :param path_temp_min: path to file with minimum temperature for each month and lat&long
    :param df_lat_long: dataframe with latitude and longitude for each country
    :param df_temp_mean_path: optional: path to dataframe where mean temperature is already computed
    :return: dataframe with mean temperature for each country and month
    """

    if df_temp_mean_path.exists():
        print(f"File {df_temp_mean_path.name} exists! Reading...")
        df_temp_mean = pd.read_csv(df_temp_mean_path, index_col=0)
        return df_temp_mean
    else:
        print(f"File {df_temp_mean_path.name} doesn't exist!\nComputing mean temperature for each country manually.")
        print("THIS MIGHT TAKE LONG!\n")

        weather_temp_max = Dataset(path_temp_max)
        weather_temp_min = Dataset(path_temp_min)

        latitudes = weather_temp_max["lat"][:]
        longitudes = weather_temp_max["lon"][:]

        df_lat_long = find_long_lat_index(latitudes, longitudes, df_lat_long)

        df_temp_mean = pd.DataFrame(index=df_lat_long.index, columns=np.arange(0, 12))

        for index, country in df_lat_long.iterrows():
            print(index)
            temp_max = weather_temp_max["tmax"][:, country["lat_idx"], country["long_idx"]]
            temp_min = weather_temp_min.variables["tmin"][:, country["lat_idx"], country["long_idx"]]

            df_temp_mean.loc[index, :] = (temp_max + temp_min) / 2

        df_temp_mean.to_csv("data/df_temp_mean.csv")

        return df_temp_mean


def drop_countries_with_no_temperature(df_mean_temperature: pd.DataFrame,
                                       df_reproduction: pd.DataFrame,
                                       df_death_ratio: pd.DataFrame):
    """
    Drops countries with no temperature

    :param df_mean_temperature: dataframe with mean temperature for each country and month
    :param df_reproduction: dataframe with reproduction coefficient for each country and day
    :param df_death_ratio: dataframe with death ratio for each country and month
    :return input dataframes with countries that are in df_mean_temperature dataframe:
    """
    NO_TEMPERATURE_VALUE = 32000
    try:
        df_mean_temperature = df_mean_temperature.loc[df_mean_temperature["0"] < NO_TEMPERATURE_VALUE, :]
    except KeyError as e:
        df_mean_temperature = df_mean_temperature.loc[df_mean_temperature[0] < NO_TEMPERATURE_VALUE, :]

    df_mean_temperature = df_mean_temperature.dropna()
    countries_with_temperature = df_mean_temperature.index.values

    # Delete countries for which we don't have measured temperature
    df_reproduction = df_reproduction[df_reproduction.index.isin(countries_with_temperature)]
    df_death_ratio = df_death_ratio[df_death_ratio.index.isin(countries_with_temperature)]

    return df_mean_temperature, df_reproduction, df_death_ratio


def normalize_reproduction_coefficient(df_reproduction: pd.DataFrame):
    """
    Calculate normalized reproduction coefficient for each country by dividing each coefficient by maximum
     coefficient per country

    :param df_reproduction: dataframe with reproduction coefficient for each country and day
    :return: dataframe with normalized reproduction coefficient for each country and day
    """

    df_max = df_reproduction.max(axis=1)
    df_reproduction_normalized = df_reproduction.div(df_max, axis=0)

    return df_reproduction_normalized


def main():
    df_deaths = read_covid_time_series_data(path="data/time_series_covid19_deaths_global.txt")
    df_recovered = read_covid_time_series_data(path="data/time_series_covid19_recovered_global.txt")
    df_confirmed = read_covid_time_series_data(path="data/time_series_covid19_confirmed_global.txt")

    df_confirmed, df_deaths, df_recovered, df_lat_long = create_long_lat_dataframe(df_confirmed,
                                                                                   df_deaths,
                                                                                   df_recovered)

    df_confirmed, df_deaths, df_recovered = clear_countries_with_no_death_data(df_confirmed, df_deaths, df_recovered)
    df_recovered = calculate_recovery_data(df_recovered, df_confirmed)

    df_active_cases = calculate_active_cases_per_day(df_confirmed, df_deaths, df_recovered)

    df_death_ratio = calculate_monthly_death_ratio(df_confirmed, df_deaths, df_recovered)
    df_reproduction = calculate_reproduction_coeff(df_active_cases)

    # Remove countries that were removed after initial clean up
    df_lat_long = df_lat_long[df_lat_long.index.isin(df_reproduction.index.values)]

    df_mean_temperature = read_terraclimate(Path("data/TerraClimate_tmax_2018.nc"),
                                            Path("data/TerraClimate_tmin_2018.nc"),
                                            df_lat_long)
    df_mean_temperature, df_reproduction, df_death_ratio = drop_countries_with_no_temperature(df_mean_temperature,
                                                                                              df_reproduction,
                                                                                              df_death_ratio)

    normalize_reproduction_coefficient(df_reproduction)


if __name__ == "__main__":
    main()
