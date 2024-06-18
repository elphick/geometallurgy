from elphick.geomet.utils.components import is_oxide, is_element, is_compositional


def test_is_element():
    res: list[str] = is_element(['SiO2', 'al2o3', 'FE', 'P'])
    assert res == ['P']

    res: dict[str, str] = is_element(['SiO2', 'al2o3', 'FE', 'P'], strict=False)
    assert res == {'FE': 'Fe', 'P': 'P'}


def test_is_oxide():
    res: list[str] = is_oxide(['SiO2', 'al2o3', 'FE'])
    assert res == ['SiO2']

    res: list[str] = is_oxide(['SiO2', 'al2o3', 'FE'], strict=False)
    assert res == {'SiO2': 'SiO2', 'al2o3': 'Al2O3'}


def test_is_compositional():
    res: list[str] = is_compositional(['SiO2', 'al2o3', 'FE', 'P'])
    assert set(res) == {'P', 'SiO2'}

    res: list[str] = is_compositional(['SiO2', 'al2o3', 'FE', 'P'], strict=False)
    assert res == {'FE': 'Fe', 'P': 'P', 'SiO2': 'SiO2', 'al2o3': 'Al2O3'}
