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


def clear_countries_with_no_recovery_data(df_confirmed: pd.DataFrame,
                                          df_deaths: pd.DataFrame,
                                          df_recovered: pd.DataFrame):
    """
    Delete countries that doesn't publish recovery data to match countries in all dataset.

    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: cleaned dataframes
    """

    recovered_countires = df_recovered.index.values
    df_deaths = df_deaths[df_deaths.index.isin(recovered_countires)]
    df_confirmed = df_confirmed[df_confirmed.index.isin(recovered_countires)]

    # One country that was death and confirmed cases wasn't publishing recovery data
    death_countries = df_deaths.index.values
    df_recovered = df_recovered[df_recovered.index.isin(death_countries)]

    return df_confirmed, df_deaths, df_recovered


def calculate_active_cases_per_day(df_confirmed: pd.DataFrame, df_deaths: pd.DataFrame, df_recovered: pd.DataFrame):
    """
    Calculates active cases per day

    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: dataframe with active cases per day
    """
    df_active_cases = pd.DataFrame(index=df_confirmed.index, columns=df_confirmed.columns)
    df_active_cases[["Lat", "Long"]] = df_confirmed[["Lat", "Long"]]
    fd_col = len(["Lat", "Long"])  # First day column number

    df_active_cases.iloc[:, fd_col:] = (df_confirmed.iloc[:, fd_col:]
                                        - df_deaths.iloc[:, fd_col:]
                                        - df_recovered.iloc[:, fd_col:])

    return df_active_cases


def create_long_lat_dataframe(df_confirmed: pd.DataFrame, df_deaths: pd.DataFrame, df_recovered: pd.DataFrame,
                              df_active_cases):
    """
    Pop longitude and latitude from dataframes for easier usage

    :param df_active_cases: daily active cases
    :param df_confirmed: dataframe with confirmed cases
    :param df_deaths: dataframe with death cases
    :param df_recovered: dataframe with recovered cases
    :return: dataframes without Lat and Long column and dataframe with Lat and Long column
    """
    df_lat_long = df_confirmed[["Lat", "Long"]]

    for df in [df_confirmed, df_deaths, df_recovered, df_active_cases]:
        df.pop("Lat")
        df.pop("Long")

    return df_confirmed, df_deaths, df_recovered, df_active_cases, df_lat_long


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
    # Delete January 2020 and January 2021
    month_end_list = pd.date_range(df_confirmed.columns[0], df_confirmed.columns[-1], freq="M")[1:]  # Delete
    month_start_list = pd.date_range(df_confirmed.columns[0], df_confirmed.columns[-1], freq="MS")[:-1]

    df_death_ratio = pd.DataFrame()
    for month_start, month_end in zip(month_start_list, month_end_list):
        df_death_ratio[month_start] = ((df_deaths[month_end] - df_deaths[month_start]) /
                                       (df_recovered[month_end] - df_recovered[month_start]))

    df_death_ratio.fillna(0)
    # TODO: What with countries that stopped publishing deaths and recovered like all regions for example China_Hubei
    # TODO: What with countries that started publishin recovery data later (for example Poland)
    plot_country("China_Hubei", df_death_ratio, "death_ratio")
    plot_country("China_Hubei", df_deaths, "deaths")
    plot_country("China_Hubei", df_recovered, "recovered")
    plot_country("China_Hubei", df_confirmed, "confirmed")
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

    for day in df_mean_active_cases.columns:
        date_range = pd.date_range(day - pd.DateOffset(days=mean_days_number - 1), day, freq="D")
        df_mean_active_cases[day] = df_active_cases[date_range].mean(axis=1)

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

    # TODO: Delete samples with less than 100 active cases

    df_mean_active_cases = calculate_mean_active_cases(df_active_cases, mean_days_number=7)

    df_reproduction = pd.DataFrame(columns=df_mean_active_cases.columns.values[reproduction_days:],
                                   index=df_mean_active_cases.index)

    for day in df_reproduction:
        reproduction_day = day - pd.DateOffset(days=reproduction_days)
        df_reproduction[day] = df_mean_active_cases[day]/df_mean_active_cases[reproduction_day]

    df_reproduction.fillna(0)

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
        df_lat_long_indexes.loc[index, "lat_idx"] = np.int((np.abs(latitudes - country_lat)).argmin())
        df_lat_long_indexes.loc[index, "long_idx"] = np.int((np.abs(longitudes - country_long)).argmin())

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
        df_temp_mean = pd.read_csv(df_temp_mean_path)
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
            for month in df_temp_mean.columns:
                print(month)
                df_temp_mean.loc[index, month] = ((weather_temp_max["tmax"]
                                                                   [month]
                                                                   [int(country["lat_idx"])]
                                                                   [int(country["long_idx"])]) +
                                                  (weather_temp_min["tmin"]
                                                                   [month]
                                                                   [int(country["lat_idx"])]
                                                                   [int(country["long_idx"])])) / 2

        df_temp_mean.to_csv("data/df_temp_mean.csv")

        return df_temp_mean


def main():
    df_deaths = read_covid_time_series_data(path="data/time_series_covid19_deaths_global.txt")
    df_recovered = read_covid_time_series_data(path="data/time_series_covid19_recovered_global.txt")
    df_confirmed = read_covid_time_series_data(path="data/time_series_covid19_confirmed_global.txt")

    df_confirmed, df_deaths, df_recovered = clear_countries_with_no_recovery_data(df_confirmed, df_deaths, df_recovered)

    df_active_cases = calculate_active_cases_per_day(df_confirmed, df_deaths, df_recovered)
    df_confirmed, df_deaths, df_recovered, df_active_cases, df_lat_long = create_long_lat_dataframe(df_confirmed,
                                                                                                    df_deaths,
                                                                                                    df_recovered,
                                                                                                    df_active_cases)
    df_death_ratio = calculate_monthly_death_ratio(df_confirmed, df_deaths, df_recovered)
    df_reproduction = calculate_reproduction_coeff(df_active_cases)

    df_mean_temperature = read_terraclimate(Path("data/TerraClimate_tmax_2018.nc"),
                                            Path("data/TerraClimate_tmin_2018.nc"),
                                            df_lat_long)


if __name__ == "__main__":
    main()
