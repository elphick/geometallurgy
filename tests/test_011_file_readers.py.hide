from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from elphick.geomet.readers import ParquetFileReader, OMFFileReader


def create_parquet(num_cols=20, num_rows=10000, num_object_vars=2) -> Path:
    import pandas as pd
    import numpy as np
    import pyarrow as pa

    # Create num_cols - num_object_vars number of float columns
    df = pd.DataFrame({f"column{i}": np.random.rand(num_rows) for i in range(num_cols - num_object_vars)})

    # Create num_object_vars number of object columns
    for i in range(num_object_vars):
        df[f"column{num_cols - num_object_vars + i}"] = ['object_data'] * num_rows

    table = pa.Table.from_pandas(df)
    file_path = Path(f'test.{num_rows}x{num_cols}.parquet')
    pq.write_table(table, file_path)
    return file_path


# create_parquet()

def test_read_parquet():
    file_path = Path('data/test.10000x20.parquet')
    df = ParquetFileReader(file_path).read(columns=['column1', 'column2'])
    assert not df.empty
    assert len(df.columns) == 2
    assert 'column1' in df.columns
    assert 'column2' in df.columns
    assert len(df) == 10000
    assert df['column1'].dtype == float
    assert df['column2'].dtype == float


def test_read_parquet_with_object_cols():
    file_path = Path('data/test.10000x20.parquet')
    df = ParquetFileReader(file_path).read(columns=['column1', 'column2', 'column18', 'column19'])
    assert not df.empty
    assert len(df.columns) == 4
    assert 'column1' in df.columns
    assert 'column2' in df.columns
    assert 'column18' in df.columns
    assert 'column19' in df.columns
    assert len(df) == 10000
    assert df['column1'].dtype == float
    assert df['column2'].dtype == float
    assert df['column18'].dtype == object
    assert df['column19'].dtype == object
    assert df['column18'].unique() == ['object_data']
    assert df['column19'].unique() == ['object_data']


def test_read_parquet_with_query():
    file_path = Path('data/test.10000x20.parquet')
    df = ParquetFileReader(file_path).read(query="column1 > 0.5")
    assert not df.empty
    assert len(df) < 10000
    assert df['column1'].dtype == float
    assert (df['column1'] > 0.5).all()
    assert len(df.columns) == 20


def test_read_parquet_with_query_and_columns():
    file_path = Path('data/test.10000x20.parquet')
    df = ParquetFileReader(file_path).read(columns=['column1', 'column2', 'column19'], query="column1 > 0.5")
    assert not df.empty
    assert len(df) < 10000
    assert df['column1'].dtype == float
    assert (df['column1'] > 0.5).all()
    assert len(df.columns) == 3
    assert 'column1' in df.columns
    assert 'column2' in df.columns
    assert 'column19' in df.columns
    assert (df['column1'] > 0.5).all()
    assert df['column19'].unique() == ['object_data']


def test_read_bm_parquet():
    file_path = Path('data/block_model_copper.parquet')
    df = ParquetFileReader(file_path).read(columns=['CU_pct'], query="CU_pct > 0.1")
    assert not df.empty
    assert len(df) < ParquetFileReader(file_path).records_in_file


def test_read_omf():
    file_path = Path('data/test_model.omf')
    df: pd.DataFrame = OMFFileReader(file_path, element='Block Model').read()
    assert not df.empty
