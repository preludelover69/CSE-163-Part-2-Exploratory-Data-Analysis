# Axel Doblado Cardenas
# CSE 163
# Autumn 2026
# CSE 163 Final Project - Part 2 (EDA)

"""
Exploratory data analysis of income and health in Washington counties
using the 2025 County Health Rankings data, joined with the Census
county boundary file for maps.
"""

import geopandas as gpd
import pandas as pd
import plotly.express as px

HEALTH_PATH = 'data/analytic_data2025_v3.csv'
SHAPES_PATH = 'data/cb_2023_us_county_20m.zip'

# Original column name -> short name
COLUMNS = {
    '5-digit FIPS Code': 'fips',
    'Name': 'county',
    'Median Household Income raw value': 'median_income',
    'Poor or Fair Health raw value': 'poor_fair_health',
    'Poor Physical Health Days raw value': 'poor_physical_days',
    'Poor Mental Health Days raw value': 'poor_mental_days',
    'Life Expectancy raw value': 'life_expectancy',
    'Uninsured raw value': 'uninsured',
    'Feelings of Loneliness raw value': 'loneliness',
    'Lack of Social and Emotional Support raw value': 'lack_support',
}

# Health measures compared between low and high income counties
MEASURES = ['poor_fair_health', 'poor_physical_days', 'poor_mental_days',
            'life_expectancy', 'uninsured', 'loneliness', 'lack_support']


def load_health(path: str) -> pd.DataFrame:
    """
    Reads the County Health Rankings CSV (skipping its second header row),
    keeps only Washington counties (not the statewide row, which has
    county code 0), keeps and renames the columns in COLUMNS, and adds an
    'income_group' column: 'High' if the county's median income is above
    the median of all counties, otherwise 'Low'.
    """
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    wa = df[(df['State Abbreviation'] == 'WA') &
            (df['County FIPS Code'] != 0)]
    wa = wa[list(COLUMNS.keys())].rename(columns=COLUMNS)
    cutoff = wa['median_income'].median()
    wa['income_group'] = wa['median_income'].apply(
        lambda income: 'High' if income > cutoff else 'Low')
    return wa.reset_index(drop=True)


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the average of each health measure for low and high income
    counties, plus the percent difference of the low income average from
    the high income average, sorted from largest difference to smallest.
    """
    result = df.groupby('income_group')[MEASURES].mean().T
    result['pct_diff'] = (result['Low'] - result['High']) / result['High']
    result['pct_diff'] = result['pct_diff'] * 100
    return result.sort_values('pct_diff', key=abs, ascending=False)


def load_merged(path: str, health: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Reads the Census county boundary file, keeps Washington counties
    (state code '53'), and joins it with the health data on the county
    FIPS code. GEOID is text in the Census file, so it is converted to an
    int to match the health data's fips column.
    """
    shapes = gpd.read_file(path)
    shapes = shapes[shapes['STATEFP'] == '53']
    shapes['fips'] = shapes['GEOID'].astype(int)
    return shapes[['fips', 'geometry']].merge(health, on='fips')


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str,
                 file_name: str) -> None:
    """
    Saves a scatter plot of y against x, one point per county, colored
    by income group, as an HTML file in the figures folder.
    """
    fig = px.scatter(df, x=x, y=y, color='income_group',
                     hover_name='county', title=title)
    fig.write_html('figures/' + file_name)


def plot_map(gdf: gpd.GeoDataFrame, column: str, title: str,
             file_name: str) -> None:
    """
    Saves a map of Washington counties colored by the given column as an
    HTML file in the figures folder.
    """
    fig = px.choropleth(gdf, geojson=gdf.geometry, locations=gdf.index,
                        color=column, hover_name='county', title=title)
    fig.update_geos(fitbounds='locations', visible=False)
    fig.write_html('figures/' + file_name)


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)

    wa = load_health(HEALTH_PATH)
    print('Shape:', wa.shape)
    print('Duplicate counties:', wa['fips'].duplicated().sum())
    print('Missing values per column:')
    print(wa.isnull().sum())
    print(wa.drop(columns='fips').describe())
    print(wa['income_group'].value_counts())

    # RQ1 and RQ3: correlations
    for measure in ['poor_fair_health', 'poor_physical_days',
                    'life_expectancy']:
        print('income vs', measure, wa['median_income'].corr(wa[measure]))
    print('loneliness vs lack_support',
          wa['loneliness'].corr(wa['lack_support']))

    # RQ2: low vs high income averages
    print(compare_groups(wa))

    # RQ4: counties with the most poor physical health days
    top = wa.sort_values('poor_physical_days', ascending=False)
    print(top[['county', 'poor_physical_days', 'income_group']].head())

    plot_scatter(wa, 'median_income', 'poor_fair_health',
                 'Median income vs. share of adults in poor or fair health',
                 'income_vs_health.html')
    plot_scatter(wa, 'lack_support', 'loneliness',
                 'Lack of social and emotional support vs. loneliness',
                 'support_vs_loneliness.html')

    # Joined dataset
    merged = load_merged(SHAPES_PATH, wa)
    print('Joined shape:', merged.shape)
    print('Missing values after join:', merged.isnull().sum().sum())
    plot_map(merged, 'median_income',
             'Median household income by county', 'map_income.html')
    plot_map(merged, 'poor_physical_days',
             'Average poor physical health days by county',
             'map_physical_days.html')


if __name__ == '__main__':
    main()