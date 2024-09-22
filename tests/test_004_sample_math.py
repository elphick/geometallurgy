import pandas as pd
import pytest

from elphick.geomet import Sample
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.data import sample_data


@pytest.fixture
def expected_data() -> pd.DataFrame:
    expected_data = sample_data(include_wet_mass=True,
                                include_dry_mass=True,
                                include_moisture=True)
    expected_data.columns = [col.lower() for col in expected_data.columns]
    expected_data.rename(columns={'wet_mass': 'mass_wet'}, inplace=True)
    return expected_data


def test_sample_split(expected_data):
    data = sample_data(include_moisture=True)
    smpl: Sample = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5)
    smpl: Stream
    pd.testing.assert_frame_equal(ref.data, comp.data)

    # test that the _node tuple values have preserved the relationship.
    # the first element of the tuple is the parent node, the second element is the child node.
    assert smpl.nodes[1] == ref.nodes[0]
    assert smpl.nodes[1] == comp.nodes[0]
    assert ref.nodes[0] == comp.nodes[0]
    assert ref.nodes[1] != comp.nodes[1]


def test_sample_add(expected_data):
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5, include_supplementary_data=True)
    smpl_new = ref.add(comp, name='sample_new', include_supplementary_data=True)
    pd.testing.assert_frame_equal(smpl.data, smpl_new.data)


def test_sample_sub(expected_data):
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5, include_supplementary_data=True)
    ref_new = smpl.sub(comp, name='ref_new', include_supplementary_data=True)
    pd.testing.assert_frame_equal(ref.data, ref_new.data)
