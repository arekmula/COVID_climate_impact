import pandas as pd
import netCDF4
import numpy as np

from pathlib import Path

from matplotlib import pyplot as plt
from netCDF4 import Dataset
from scipy.stats import f_oneway, normaltest, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower


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
    # Delete "_" from country with no region
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

    # plot_country("Netherlands", df_death_ratio, "death_ratio")
    # plot_country("Estonia", df_deaths, "Estonia deaths")
    # plot_country("France", df_recovered, "France recovered")
    # plot_country("Netherlands", df_confirmed, "confirmed")
    #
    # plt.show()

    return df_death_ratio


def calculate_mean_active_cases(df_active_cases: pd.DataFrame, mean_days_number=7, active_cases_threshold=100):
    """
    Calculate mean of active cases from last mean_days_number

    :param active_cases_threshold: threshold of daily active cases
    :param df_active_cases: dataframe with active cases
    :param mean_days_number: number of days to calculate mean
    :return: dataframe with mean active cases
    """

    df_mean_active_cases = pd.DataFrame(columns=df_active_cases.columns.values[mean_days_number:],
                                        index=df_active_cases.index)

    df_active_cases_copy = df_active_cases.copy()
    # If active case in day is less than 100, then put NAN in that place. This will ensure, that we will skip the data
    # that has less than 100 cases per day in calculating mean active cases in the "mean_days_number" period.
    df_active_cases_copy[df_active_cases_copy < active_cases_threshold] = np.nan

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
                      df_temp_mean_path=Path("df_temp_mean.csv")):
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

        df_temp_mean.to_csv("df_temp_mean.csv")

        return df_temp_mean


def drop_countries_with_no_temperature(df_mean_temperature: pd.DataFrame,
                                       df_reproduction: pd.DataFrame):
    """
    Drops countries with no temperature

    :param df_mean_temperature: dataframe with mean temperature for each country and month
    :param df_reproduction: dataframe with reproduction coefficient for each country and day
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

    return df_mean_temperature, df_reproduction


def normalize_reproduction_coefficient(df_reproduction: pd.DataFrame):
    """
    Calculate normalized reproduction coefficient for each country by calculating mean reproduction per month and
    dividing each coefficient by country's maximum coefficient

    :param df_reproduction: dataframe with reproduction coefficient for each country and day
    :return: dataframe with normalized reproduction coefficient for each country and day
    """

    df_reproduction_mean = df_reproduction.resample(rule="M", axis=1).mean()
    df_max = df_reproduction_mean.max(axis=1)
    df_reproduction_normalized = df_reproduction_mean.div(df_max, axis=0)

    return df_reproduction_normalized


def convert_mean_temperature_to_ranges(df_mean_temperature: pd.DataFrame):
    """
    Convert mean temperatures per month to discrete values [<0; 0-10, 10-20, 20-30, 30>]

    :param df_mean_temperature: dataframe with mean temperature per month and country
    :return: dataframe with temperature saved in categorical data
    """

    df_mean_temperature_converted = df_mean_temperature.copy()
    temperature_ranges_masks = [(df_mean_temperature < 0),
                                ((df_mean_temperature >= 0) & (df_mean_temperature < 10)),
                                ((df_mean_temperature >= 10) & (df_mean_temperature < 20)),
                                ((df_mean_temperature >= 20) & (df_mean_temperature < 30)),
                                (df_mean_temperature >= 30)]

    for idx, temperature_mask in enumerate(temperature_ranges_masks):
        df_mean_temperature_converted[temperature_mask] = idx

    return df_mean_temperature_converted


def create_reproduction_temperature_dict(df_mean_temperature_converted: pd.DataFrame,
                                         df_reproduction_normalized: pd.DataFrame):
    """
    Creates dictionary with temperature bins as keys. Each temperature bin has it list of reproduction
     coefficients

    :param df_mean_temperature_converted: Dataframe with mean_temperature converted to categorical data
    :param df_reproduction_normalized: Dataframe with normalized reproduction coefficient per country
    :return: Dictionary with temperature bins as keys. Each temperature bin has it list of reproduction
    coefficients
    """

    # Convert full DateTime from index to only month name. Move January 2021 to first column to match months order in
    # df_reproduction and df_mean_temperature dataframes
    df_reproduction_normalized.columns = df_reproduction_normalized.columns.month
    df_reproduction_normalized = df_reproduction_normalized.reindex(sorted(df_reproduction_normalized.columns), axis=1)

    # Add 1 to all indexes to match month indexes. So January -> 1, February -> 2 etc.
    df_mean_temperature_converted.columns = np.arange(1, 13)

    temperature_bins = ["<0", "0-10", "10-20", "20-30", ">30"]
    temperature_reproduction_coeff = {0: [], 1: [], 2: [], 3: [], 4: []}

    for country in df_reproduction_normalized.index.values:
        for month in df_reproduction_normalized.columns.values:
            temperature_bin = df_mean_temperature_converted.loc[country, month]
            reproduction_value = df_reproduction_normalized.loc[country, month]
            if not pd.isna(reproduction_value):
                temperature_reproduction_coeff[temperature_bin].append(reproduction_value)

    for bin_number, bin_key in zip(list(temperature_reproduction_coeff.keys()), temperature_bins):
        temperature_reproduction_coeff[bin_key] = temperature_reproduction_coeff.pop(bin_number)

    return temperature_reproduction_coeff


def check_temperature_hypothesis(temperature_reproduction_coeff: dict, alpha=0.05):
    """
    Check hypothesis that temperature has impact on COVID-19 transmission

    :param temperature_reproduction_coeff: Dictionary with temperature bins as keys. Each temperature bin has it list of reproduction
    coefficients
    :param alpha: significance level
    :return:
    """

    print("\n\n########################################################################")
    print("########################################################################")
    print("Czy temperatura otoczenia istotnie wpływa na szybkość rozprzestrzeniania się wirusa?")
    print("Hipoteza H_0: Temperatura otoczenia nie wpływa na szybkość rozprzestrzeniania się wirusa")
    print("Hipoteza H_1: Temperatura otoczenia wpływa na szybkość rozprzestrzeniania się wirusa")
    print(f"Zakładany przedział ufności: {1 - alpha}")

    # Unpack the data from the dictionary
    data_less_0 = temperature_reproduction_coeff["<0"]
    data_0_10 = temperature_reproduction_coeff["0-10"]
    data_10_20 = temperature_reproduction_coeff["10-20"]
    data_20_30 = temperature_reproduction_coeff["20-30"]
    data_greater_30 = temperature_reproduction_coeff[">30"]

    # Checking if data distribution is normal
    print("\nSprawdzanie czy testowane zbiory mają rozkład normalny:")
    for data_bin in temperature_reproduction_coeff.keys():
        stat, p = normaltest(temperature_reproduction_coeff[data_bin])
        if p < alpha:
            print(f"Odrzucamy hipotezę 0, że rozkład {data_bin} nie jest normalny. Przyjmujemy hipotezę alternatywną,"
                  f" że rozkład jest normalny")

    # Checking if standard deviation is close between data
    print("\nSprawdzanie czy testowane zbiory mają zbliżoną wariancję/odchylenie standardowe")
    std_devs = []
    for data_bin in temperature_reproduction_coeff.keys():
        std = np.std(temperature_reproduction_coeff[data_bin])
        std_devs.append(std)
    print(f"Testowane zbiory mają zbliżone odchylenia standardowe: {std_devs}")

    f_value, p_value = f_oneway(data_less_0, data_0_10, data_10_20, data_20_30, data_greater_30)
    print(f"\nF-stat {f_value}, Prawdopodobieństwo testowe: {p_value:.9f}")
    if p_value < alpha:
        print(f"Prawdopodobieństwo testowe {p_value:.9f} < poziom istotności {alpha}")
        print(f"Istnieje istotna różnica! Można wykluczyć hipotezę zerową.")
        print("\nAnaliza post-hoc")
        print(pairwise_tukeyhsd(
            np.concatenate([data_less_0, data_0_10, data_10_20, data_20_30, data_greater_30]),
            np.concatenate([["data_less_0"] * len(data_less_0), ["data_0_10"] * len(data_0_10),
                            ["data_10_20"] * len(data_10_20), ["data_20_30"] * len(data_20_30),
                            ["data_greater_30"] * len(data_greater_30)])))

        print("Istnieje istotna różnica między zbiorami \"0-10\" i \"20-30\", zbiorami \"10-20\" i \"20-30\" oraz"
              f" zbiorami \"10-20\" i \">30\" dla poziomu ufności {1 - alpha}. \nMożna dla tych zbiorów odrzucić"
              f" hipotezę H0 i przyjąć hipotezę alterantywną, że temperatura otoczenia wpływa na szybkość"
              f" rozprzestrzeniania się wirusa."
              f"\nDla pozostałych zbiorów mamy za mało próbek aby odrzucić, bądź potwierdzić hipotezę 0.")

        # Check power of test
        data_pairs = [(data_0_10, data_20_30), (data_10_20, data_20_30), (data_10_20, data_greater_30)]
        data_pairs_names = [("0-10", "20-30"), ("10-20", "20-30"), ("10-20", ">30")]
        analysis = TTestIndPower()
        print("Obliczanie mocy testu dla zbiorów z istotną różnicą:")
        for pair, pair_names in zip(data_pairs, data_pairs_names):
            effect = (np.mean(pair[0]) - np.mean(pair[1])) / ((np.std(pair[0]) + np.std(pair[1])) / 2)
            result = analysis.solve_power(effect, power=None, nobs1=len(pair[0]), ratio=1.0, alpha=alpha)
            print(f"Dla zbiorów {pair_names[0]} i {pair_names[1]} możemy przyjąć hipotezę alternatywną z"
                  f" prawdopodobieństwem: {result:.4f}")


def check_deaths_difference_europe_chi2(df_confirmed: pd.DataFrame, df_deaths: pd.DataFrame, european_countries: list,
                                        alpha=0.05):
    """
    Checks if there's a big difference in deaths in european
    
    :param alpha: significance level
    :param df_confirmed: dataframe with sum of confirmed cases per country
    :param df_deaths: dataframe with sum of death cases per country
    :param european_countries: list of european countries
    :return: 
    """

    print("\n\n\n\n########################################################################")
    print("########################################################################")
    print("Czy między poszczególnymi krajami w Europie istnieją różnice w śmiertelności z powodu COVID-19?"
          " \nWersja z testem chi2")
    print("Hipoteza H_0: Pomiędzy poszczególnymi krajami w Europie nie ma różnicy w śmiertelności z powodu COVID-19")
    print("Hipoteza H_1: Pomiędzy poszczególnymi krajami w Europie jest różnica w śmiertelności z powodu COVID-19")
    print(f"Zakładany przedział ufności: {1 - alpha}")

    df_confirmed_european: pd.DataFrame = df_confirmed.loc[european_countries, :]
    df_deaths_european: pd.DataFrame = df_deaths.loc[european_countries, :]

    df_deaths_vs_confirmed = pd.DataFrame(index=df_confirmed_european.index, columns=["confirmed", "deaths"])

    # Save last sum of confirmed and sum of death cases to dataframe
    df_deaths_vs_confirmed["confirmed"] = df_confirmed_european.iloc[:, -1]
    df_deaths_vs_confirmed["deaths"] = df_deaths_european.iloc[:, -1]

    np_deaths_vs_confirmed = df_deaths_vs_confirmed.to_numpy()

    chi2, p_val, df, _ = chi2_contingency(np_deaths_vs_confirmed)
    print(f"\nPrawdopodobieństwo testowe: {p_val}, chi2: {chi2:.0f}, Liczba stopni swobody: {df}")
    if p_val < alpha:
        print(f"Prawdopodobieństwo testowe {p_val} < poziom istotności {alpha}")
        analysis = TTestIndPower()
        effect = ((np.mean(df_deaths_vs_confirmed["confirmed"]) - np.mean(df_deaths_vs_confirmed["deaths"])) /
                  ((np.std(df_deaths_vs_confirmed["confirmed"]) + np.std(df_deaths_vs_confirmed["deaths"])) / 2))
        result = analysis.solve_power(effect, power=None, nobs1=len(df_deaths_vs_confirmed["confirmed"]), ratio=1.0,
                                      alpha=alpha)
        print(f"Istnieje istotna różnica! Można wykluczyć hipotezę zerową oraz przyjąć z prawdopodobieństwem {result}"
              f" hipotezę alternatywną, że istnieje różnica w śmiertelności pomiędzy poszczególnymi krajami w Europie")


def check_deaths_difference_europe_anova(df_death_ratio: pd.DataFrame, european_countries: list, alpha=0.05):
    """
    Check if there's big difference in deaths in european countries. ANOVA test

    :param european_countries: list of european countries from dataframe
    :param alpha: significance level
    :param df_death_ratio: Dataframe with death ratio per month and country
    :return:
    """

    print("\n\n\n\n########################################################################")
    print("########################################################################")
    print("Czy między poszczególnymi krajami w Europie istnieją różnice w śmiertelności z powodu COVID-19?"
          " \nWersja z testem ANOVA")
    print("Hipoteza H_0: Pomiędzy poszczególnymi krajami w Europie nie ma różnicy w śmiertelności z powodu COVID-19")
    print("Hipoteza H_1: Pomiędzy poszczególnymi krajami w Europie jest różnica w śmiertelności z powodu COVID-19")
    print(f"Zakładany przedział ufności: {1 - alpha}")

    df_death_ratio_europe: pd.DataFrame = df_death_ratio.loc[european_countries, :]

    # Delete Countries with inf and -inf in Dataframe
    df_death_ratio_europe = df_death_ratio_europe[~df_death_ratio_europe.isin([np.inf, -np.inf]).any(1)]

    # France and Estonia have negative values in death/ratio coefficients
    # France had some mismatch in recovery data in July -> uncomment lines in function calculate_monthly_death_ratio
    # to plot this
    # Estonia had some mismatch in death data in August -> uncomment lines in function calculate_monthly_death_ratio to
    # plot this
    df_death_ratio_europe = df_death_ratio_europe.drop(["France", "Estonia"], axis=0)

    # Checking if data distribution is normal
    print("\nSprawdzanie czy testowane zbiory mają rozkład normalny")
    print("Uwaga: Test kurtozy w pakiecie scipy.stats jest prawidłowy tylko dla więcej niż 20 próbek. My mamy tylko"
          " 11 próbek z ostatnich 11 miesięcy")
    stat, p = normaltest(df_death_ratio_europe, axis=1)
    if not (p < alpha).all():
        countries_indexes_to_drop = np.where(p>alpha)
        print(f"Dla państw {df_death_ratio_europe.index.values[countries_indexes_to_drop]} możemy przyjąć hipotezę H0,"
              f"że rozkład nie jest normalny. Państwa zostają odrzucone w dalszej analizie")
        df_death_ratio_europe = df_death_ratio_europe.loc[p<alpha]

    print("Dla pozostałych państw odrzuczamy hipotezę 0, że rozkład dla któregoś z państw nie jest normalny."
          " Przyjmujemy hipotezę alternatywną, że jest normalny")

    # Checking if standard deviation is close between data
    print("\nSprawdzanie czy testowane zbiory mają zbliżoną wariancję/odchylenie standardowe")
    df_std_dev: pd.DataFrame = df_death_ratio_europe.std(axis=1)
    df_std_dev_med = df_std_dev.median()
    countries_to_drop = ["Albania", "Austria", "Denmark", "Luxembourg", "Moldova", "Poland", "Portugal", "Slovakia",
                         "Switzerland"]
    print(f"Dla państw: {countries_to_drop} odchylenie standardowe różni się o rząd wielkości od mediany odchyleń"
          f" {df_std_dev_med:.2f}. \nPaństwa te zostają odrzucone w dalszej analizie.")
    df_death_ratio_europe = df_death_ratio_europe.drop(countries_to_drop)

    np_death_ratio_europe = df_death_ratio_europe.to_numpy()
    death_ratio_europe_list = np_death_ratio_europe.tolist()

    f_value, p_value = f_oneway(*death_ratio_europe_list)
    print(f"\nF-stat {f_value:.4f}, Prawdopodobieństwo testowe: {p_value:.4f}")
    if p_value < alpha:
        print(f"Prawdopodobieństwo testowe {p_value:.9f} < poziom istotności {alpha}")
        print(f"Istnieje istotna różnica! Można wykluczyć hipotezę zerową.")
        print("\nAnaliza post-hoc ...")
    else:
        print(f"Prawdopodobieństwo testowe {p_value:.4f} > poziom istotności {alpha}")
        print(f"Nie można wykluczyć hipotezy H_0, że pomiędzy poszczególnymi krajami w Europie nie ma różnicy w"
              f" śmiertelności z powodu COVID-19!")


def main():
    df_deaths = read_covid_time_series_data(path="time_series_covid19_deaths_global.txt")
    df_recovered = read_covid_time_series_data(path="time_series_covid19_recovered_global.txt")
    df_confirmed = read_covid_time_series_data(path="time_series_covid19_confirmed_global.txt")

    df_confirmed, df_deaths, df_recovered, df_lat_long = create_long_lat_dataframe(df_confirmed=df_confirmed,
                                                                                   df_deaths=df_deaths,
                                                                                   df_recovered=df_recovered)

    df_confirmed, df_deaths, df_recovered = clear_countries_with_no_death_data(df_confirmed=df_confirmed,
                                                                               df_deaths=df_deaths,
                                                                               df_recovered=df_recovered)

    df_recovered = calculate_recovery_data(df_recovered, df_confirmed)

    df_active_cases = calculate_active_cases_per_day(df_confirmed=df_confirmed,
                                                     df_deaths=df_deaths,
                                                     df_recovered=df_recovered)

    df_death_ratio = calculate_monthly_death_ratio(df_confirmed=df_confirmed,
                                                   df_deaths=df_deaths,
                                                   df_recovered=df_recovered)
    df_reproduction = calculate_reproduction_coeff(df_active_cases)

    # Remove countries that were removed after initial clean up
    df_lat_long = df_lat_long[df_lat_long.index.isin(df_reproduction.index.values)]

    df_mean_temperature = read_terraclimate(Path("TerraClimate_tmax_2018.nc"),
                                            Path("TerraClimate_tmin_2018.nc"),
                                            df_lat_long)
    df_mean_temperature, df_reproduction = drop_countries_with_no_temperature(df_mean_temperature,
                                                                              df_reproduction)

    df_reproduction_normalized = normalize_reproduction_coefficient(df_reproduction)

    df_mean_temperature_converted = convert_mean_temperature_to_ranges(df_mean_temperature)

    temperature_reproduction_coeff = create_reproduction_temperature_dict(df_mean_temperature_converted,
                                                                          df_reproduction_normalized)

    check_temperature_hypothesis(temperature_reproduction_coeff=temperature_reproduction_coeff)

    # List of european countries needed for next 2 tests
    european_countries = ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
                          "Croatia", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Hungary",
                          "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
                          "Malta", "Moldova", "Monaco", "Netherlands", "Norway", "North Macedonia", "Poland",
                          "Portugal", "Romania", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
                          "Sweden", "Switzerland", "Ukraine", "United Kingdom"]

    check_deaths_difference_europe_chi2(df_confirmed=df_confirmed,
                                        df_deaths=df_deaths,
                                        european_countries=european_countries)

    check_deaths_difference_europe_anova(df_death_ratio, european_countries=european_countries)


if __name__ == "__main__":
    main()
