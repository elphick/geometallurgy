from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandera as pa
import pandas as pd
from omf import OMFReader
import concurrent.futures



class FileValidator(ABC):
    def __init__(self, file_path: Path, schema=None):
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        self.file_path = file_path
        self.schema = schema or {}

    @abstractmethod
    def validate(self):
        pass


import pandera.io


class ParquetFileValidator(FileValidator):
    """
    Validate a Parquet file against a Pandera schema
    """

    def __init__(self, file_path: Path, schema_path: Path):
        """
        Initialize the Parquet file validator
        Args:
            file_path: The path to the Parquet file
            schema_path: The path to the YAML file containing the schema
        """
        schema = pandera.io.from_yaml(schema_path)
        super().__init__(file_path, schema)
        self.store: pd.HDFStore = pd.HDFStore('coerced.h5')

    def validate(self, max_workers: int = 20):
        super().validate()
        columns = list(self.schema.columns.keys())
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._validate_column, column): column for column in columns}
            for future in concurrent.futures.as_completed(futures):
                column = futures[future]
                try:
                    future.result()
                except Exception as e:
                    raise ValueError(f"Invalid data in column {column}: {e}")

    def _validate_column(self, column):
        try:
            df = pd.read_parquet(self.file_path, columns=[column])
            column_schema = {column: self.schema.columns[column]}
            schema = pa.DataFrameSchema(column_schema)
            coerced_df = schema.validate(df)
            file_stem = self.file_path.stem  # get the stem of the original file
            hdf_key = f"{file_stem}/{column}"  # create a hierarchical key using the file stem and column name
            self.store.put(hdf_key, coerced_df, format='table')
            # if Path('coerced.h5').exists():
            #     coerced_df.to_hdf('coerced.h5', key=hdf_key, mode='a')
            # else:
            #     coerced_df.to_hdf('coerced.h5', key=hdf_key, mode='w')
        except Exception as e:
            raise ValueError(f"Invalid Parquet file or schema: {e}")
class OMFFileValidator(FileValidator):

    def __init__(self, file_path, element: str, schema=None):
        """
        Initialize the Parquet file validator
        Args:
            file_path: The path to the OMF file
            element: the element in the OMF file to be validated. E.g. 'Block Model'
            schema: The pandera schema to validate the file against
        """
        super().__init__(file_path, schema)

        # check that the element provided is a valid VolumeElement in the OMF file.
        elements = OMFReader(file_path).get_project().elements
        if element not in elements:
            raise ValueError(f"Element '{element}' not found in the OMF file: {file_path}")
        elif elements[element].__class__.__name__ != 'VolumeElement':
            raise ValueError(f"Element '{element}' is not a VolumeElement in the OMF file: {file_path}")

        self.element = element

    def validate(self):
        super().validate()
        try:
            OMFReader(self.file_path)
        except Exception as e:
            raise ValueError(f"Invalid OMF file: {e}")
