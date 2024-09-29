# from elphick.geomet import Sample, MassComposition, IntervalSample, Stream
from elphick.geomet.base import MassComposition
from elphick.geomet.block_model import BlockModel
from elphick.geomet.datasets import Downloader
from elphick.geomet.interval_sample import IntervalSample
from elphick.geomet.sample import Sample
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.data import sample_data

from fixtures import omf_model_path


def test_inheritance_sample():
    data = sample_data(include_moisture=True)
    obj: Sample = Sample(data=data, name='sample')
    assert isinstance(obj, Sample)
    assert isinstance(obj, MassComposition)
    assert not isinstance(obj, Stream)

    # check the object and result are a Stream after math operations
    obj_res: Stream = obj + obj

    assert isinstance(obj, Sample)
    assert isinstance(obj, MassComposition)
    assert isinstance(obj, Stream)
    assert isinstance(obj_res, Sample)
    assert isinstance(obj_res, MassComposition)
    assert isinstance(obj_res, Stream)


def test_inheritance_interval_sample():
    data = Downloader().load_data(datafile='iron_ore_sample_A072391.zip', show_report=False)
    obj: IntervalSample = IntervalSample(data=data, name='interval_sample')
    assert isinstance(obj, IntervalSample)
    assert isinstance(obj, MassComposition)
    assert not isinstance(obj, Stream)

    # check the object and result are a IntervalSample after math operations
    obj_res = obj + obj
    assert isinstance(obj, IntervalSample)
    assert isinstance(obj, MassComposition)
    assert isinstance(obj, Stream)
    assert isinstance(obj_res, IntervalSample)
    assert isinstance(obj_res, MassComposition)
    assert isinstance(obj_res, Stream)


def test_inheritance_block_model(omf_model_path):
    obj: BlockModel = BlockModel.from_omf(omf_filepath=omf_model_path, columns=['CU_pct'])
    assert isinstance(obj, BlockModel)
    assert isinstance(obj, MassComposition)
    assert not isinstance(obj, Stream)

    # check the object and result are a BlockModel after math operations
    obj_res = obj + obj
    assert isinstance(obj, BlockModel)
    assert isinstance(obj, MassComposition)
    assert isinstance(obj, Stream)
    assert isinstance(obj_res, BlockModel)
    assert isinstance(obj_res, MassComposition)
    assert isinstance(obj_res, Stream)
