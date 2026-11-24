# Axel Doblado Cardenas
# CSE 163
# Autumn 2026
# CSE 163 Final Project - Part 2 (EDA)

"""
Tests for eda.py. test_data/small_counties.csv has the same layout as the
real data with a Washington statewide row, 4 Washington counties with
incomes 40k, 50k, 60k, 70k, and 1 Oregon county.
test_data/small_shapes.geojson has shapes for 3 of those counties
(Alpha, Beta, Gamma) plus one extra Washington county and one Oregon
county.
"""

import math

import eda

SMALL_CSV = 'test_data/small_counties.csv'
SMALL_SHAPES = 'test_data/small_shapes.geojson'


def test_load_health() -> None:
    """
    Only the 4 Washington counties should be kept. The median income is
    55k, so Alpha and Beta are Low and Gamma and Delta are High.
    """
    wa = eda.load_health(SMALL_CSV)
    assert list(wa['county']) == ['Alpha County', 'Beta County',
                                  'Gamma County', 'Delta County']
    assert list(wa['income_group']) == ['Low', 'Low', 'High', 'High']
    assert list(wa.columns) == list(eda.COLUMNS.values()) + ['income_group']


def test_compare_groups() -> None:
    """
    Poor or fair health: Low = (0.3, 0.2) average 0.25, High = (0.2, 0.1)
    average 0.15, so pct_diff = (0.25 - 0.15) / 0.15 * 100 = 66.67.
    Every other measure is 0.1 everywhere, so its pct_diff is 0.
    """
    result = eda.compare_groups(eda.load_health(SMALL_CSV))
    assert math.isclose(result.loc['poor_fair_health', 'Low'], 0.25)
    assert math.isclose(result.loc['poor_fair_health', 'High'], 0.15)
    assert math.isclose(result.loc['poor_fair_health', 'pct_diff'],
                        200 / 3)
    assert result.index[0] == 'poor_fair_health'
    assert math.isclose(result.loc['loneliness', 'pct_diff'], 0,
                        abs_tol=1e-9)


def test_load_merged() -> None:
    """
    Only Alpha, Beta, and Gamma are in both files, so the join should
    have 3 rows with a geometry column.
    """
    merged = eda.load_merged(SMALL_SHAPES, eda.load_health(SMALL_CSV))
    assert sorted(merged['county']) == ['Alpha County', 'Beta County',
                                        'Gamma County']
    assert 'geometry' in merged.columns


def test_real_data() -> None:
    """
    Washington has 39 counties, each listed once, and all 39 should
    match a county shape.
    """
    wa = eda.load_health(eda.HEALTH_PATH)
    assert len(wa) == 39
    assert wa['fips'].duplicated().sum() == 0
    assert len(eda.load_merged(eda.SHAPES_PATH, wa)) == 39


def main():
    test_load_health()
    test_compare_groups()
    test_load_merged()
    test_real_data()
    print('All tests passed!')


if __name__ == '__main__':
    main()