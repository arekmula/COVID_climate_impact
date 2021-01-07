import pandas as pd


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


def main():
    df_deaths = read_covid_time_series_data(path="time_series_covid19_deaths_global.txt")
    df_recovered = read_covid_time_series_data(path="time_series_covid19_recovered_global.txt")
    df_confirmed = read_covid_time_series_data(path="time_series_covid19_confirmed_global.txt")

    df_confirmed, df_deaths, df_recovered = clear_countries_with_no_recovery_data(df_confirmed, df_deaths, df_recovered)
    df_active_cases = calculate_active_cases_per_day(df_confirmed, df_deaths, df_recovered)


if __name__ == "__main__":
    main()
