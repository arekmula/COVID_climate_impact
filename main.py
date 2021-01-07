import pandas as pd
from matplotlib import pyplot as plt


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
    plt.show()

    return df_death_ratio


def main():
    df_deaths = read_covid_time_series_data(path="time_series_covid19_deaths_global.txt")
    df_recovered = read_covid_time_series_data(path="time_series_covid19_recovered_global.txt")
    df_confirmed = read_covid_time_series_data(path="time_series_covid19_confirmed_global.txt")

    df_confirmed, df_deaths, df_recovered = clear_countries_with_no_recovery_data(df_confirmed, df_deaths, df_recovered)

    # df_active_cases = calculate_active_cases_per_day(df_confirmed, df_deaths, df_recovered)
    df_confirmed, df_deaths, df_recovered, df_lat_long = create_long_lat_dataframe(df_confirmed,
                                                                                   df_deaths,
                                                                                   df_recovered)
    df_death_ratio = calculate_monthly_death_ratio(df_confirmed, df_deaths, df_recovered)


if __name__ == "__main__":
    main()
