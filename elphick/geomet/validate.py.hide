"""
Classes to support validation of block model files.
"""

import logging
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from elphick.geomet.readers import ParquetFileReader, OMFFileReader
from elphick.geomet.utils.components import is_compositional


#
# class FileValidator(ABC):
#     def __init__(self, file_path: Path, schema_path: Optional[Path] = None,
#                  lazy_validation: bool = True,
#                  negative_to_nan_threshold: float = 0):
#         if not file_path.exists():
#             raise ValueError(f"File does not exist: {file_path}")
#         self._logger = logging.getLogger(self.__class__.__name__)
#         self.file_path = file_path
#         self.schema_path = schema_path
#         self.schema: DataFrameSchema = DataFrameSchema({}) if schema_path is None else pandera.io.from_yaml(schema_path)
#         self.lazy_validation = lazy_validation
#         self.negative_to_nan_threshold = negative_to_nan_threshold
#
#         self.report: Optional[dict] = None
#
#     @abstractmethod
#     def validate(self):
#         pass
#
#     def create_schema_file(self, schema_output_path: Path):
#         """
#         Create an inferred schema file from the file being validated
#         Args:
#             schema_output_path: The output path for the schema file
#
#         Returns:
#
#         """
#
#         df = self.read_column()
#
#         with open(schema_output_path, 'w') as f:
#             yaml.dump(self.schema.to_yaml(), f)


class BaseProcessor(ABC):
    """
    To support columnar processing of large datasets, the BaseProcessor class provides a framework for processing
    data by column. The process method will process the data by column if a file_path is provided, or the entire
    dataset if data is provided.
    """

    def __init__(self, file_path: Optional[Path] = None, data: Optional[pd.DataFrame] = None, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        if file_path is None and data is None:
            raise ValueError("Either file_path or data must be provided.")
        self.file_path = file_path
        self.data = data
        self.temp_files = []

        if self.file_path.suffix == '.parquet':
            self.reader: ParquetFileReader = ParquetFileReader(self.file_path)
        elif self.file_path.suffix == '.omf':
            self.reader: OMFFileReader = OMFFileReader(self.file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

    @property
    def composition_variables(self) -> list[str]:
        """
        Detect columns that contain composition data

        Returns:
            A list of column names that contain composition data
        """
        res = None
        if self.reader.variables_in_file:
            res = list(is_compositional(self.reader.variables_in_file, strict=False).keys())
        return res

    def process(self, num_workers: Optional[int] = 1, **kwargs):
        if self.data is None:
            with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='geomet-processor') as executor:
                futures = {executor.submit(self._process_variable, variable, **kwargs): variable for variable in
                           self.reader.variables_in_file}
                results = {}
                for future in as_completed(futures):
                    variable = futures[future]
                    try:
                        results[variable] = future.result()
                    except Exception as exc:
                        print(f'{variable} generated an exception: {exc}')
        else:
            results = self._process_data()
        return results

    @abstractmethod
    def _process_variable(self, column, **kwargs):
        pass

    @abstractmethod
    def _process_data(self):
        pass


class PreProcessor(BaseProcessor):
    def __init__(self, file_path: Optional[Path] = None, data: Optional[pd.DataFrame] = None, **kwargs):
        """
        Preprocess data before validation.
        For large datasets where memory may be constrained, file_path will provide processing by columns.
        If data is provided, the entire dataset already in memory will be processed.
        Args:
            file_path: The optional path to the file to be preprocessed.
            data: The optional DataFrame to be preprocessed.
        """

        super().__init__(file_path, data, **kwargs)

    def process(self, negative_to_nan_threshold: Optional[float] = -1,
                not_detected_assays_threshold: Optional[float] = 0.5,
                max_workers=1):
        super().process(max_workers=max_workers, negative_to_nan_threshold=negative_to_nan_threshold,
                        not_detected_assays_threshold=not_detected_assays_threshold)

    def _process_variable(self, column, **kwargs):
        data = pd.read_parquet(self.file_path, columns=[column])
        processed_data = self._process_data(data)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        processed_data.to_parquet(temp_file.name)
        self.temp_files.append(temp_file)

    def _process_data(self) -> pd.DataFrame:
        # Preprocessing logic here
        return data


class Validator(BaseProcessor):
    def __init__(self, file_path: Optional[Path] = None, data: Optional[pd.DataFrame] = None, **kwargs):
        """
        Validate the data using a pandera schema.
        For large datasets where memory may be constrained file_path will provide processing by columns.
        If data is provided, the entire dataset already in memory will be processed.
        Args:
            file_path: The optional path to the file to be preprocessed.
            data: The optional DataFrame to be preprocessed.
        """
        super().__init__(file_path, data, **kwargs)

    def process(self):
        if self.data is None:
            columns = get_parquet_columns(self.file_path)
            with ThreadPoolExecutor() as executor:
                for column in columns:
                    executor.submit(self._process_variable, column)
        else:
            self._process_data()

    def _process_variable(self, column):
        data = pd.read_parquet(self.file_path, columns=[column])
        processed_data = self._process_data(data)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        processed_data.to_parquet(temp_file.name)
        self.temp_files.append(temp_file)

    def _process_data(self, data):
        # Validation logic here
        return data
