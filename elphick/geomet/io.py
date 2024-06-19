import json
import logging
import tokenize
from abc import abstractmethod, ABC
from io import StringIO
from pathlib import Path
from typing import Optional

import pyarrow as pa
import os

import numpy as np
import pandas as pd
from omf import OMFReader, VolumeGridGeometry
import pyarrow.parquet as pq
from pandera import DataFrameSchema


class BaseReader(ABC):

    def __init__(self, file_path: Path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.file_path: Path = file_path
        self.variables_in_file: list[str] = []
        self.records_in_file: int = 0

    @staticmethod
    def _parse_query_columns(query) -> list[str]:
        # Create a list to store the column names
        column_names = []

        # Tokenize the query string
        for token in tokenize.generate_tokens(StringIO(query).readline):
            token_type, token_string, _, _, _ = token

            # If the token is a name, and it's not a built-in Python name, add it to the list
            if token_type == tokenize.NAME and token_string not in __builtins__:
                column_names.append(token_string)

        return column_names

    @abstractmethod
    def read(self, columns: Optional[list[str]] = None, query: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_index(self) -> pd.Index:
        pass

    def validate(self, schema_file: Path, data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Validate using a pandera schema

        This method does not leverage multiprocessing, and loads the entire dataframe into memory.
        Args:
            schema_file: The path to the schema yaml file
            data: The data to validate, if not provided, the underlying read method will be called.
        Returns:
            The coerced DataFrame after validation
        """
        import pandera as pa
        schema: DataFrameSchema = pa.DataFrameSchema.from_yaml(schema_file)
        if data:
            df = data
        else:
            df = self.read()
        schema.validate(df, lazy=True, inplace=True)
        return df

    def preprocess(self, negative_to_nan_threshold: Optional[float] = -1,
                   not_detected_assays_threshold: Optional[float] = 0.5) -> pd.DataFrame:
        """
        Preprocess the data by managing negative values.
        Args:
            negative_to_nan_threshold: Values below this threshold will be replaced with NaN
            not_detected_assays_threshold: Values above this threshold will be replaced with half the absolute value

        Returns:
            The preprocessed DataFrame, with no negatives and no values above the not_detected_assays_threshold.

        """
        if negative_to_nan_threshold > 0:
            raise ValueError("The negative_to_nan_threshold must be less than or equal to zero or None.")
        if not_detected_assays_threshold > 0:
            raise ValueError("The not_detected_assays_threshold must be less than or equal to zero or None")

        df = self.read()

        # detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if negative_to_nan_threshold:
            df.loc[df[numeric_cols] < negative_to_nan_threshold, numeric_cols] = np.nan
        if not_detected_assays_threshold:
            mask = (df[numeric_cols] > not_detected_assays_threshold) and (df[numeric_cols] < 0)
            df.loc[mask, numeric_cols] = np.abs(df.loc[mask, numeric_cols]) / 2
        return df


class ParquetFileReader(BaseReader):
    """
    Read a Parquet file
    """

    def __init__(self, file_path: Path):
        """
        Initialize the parquet reader.  While not enforced, it is expected that the file is indexed by x, y, z, or
        x, y, z, dx, dy, dz
        Args:
            file_path: The path to the Parquet file.
        """
        super().__init__(file_path)
        self.variables_in_file = self._get_parquet_columns()
        self.records_in_file = self._get_parquet_length()

    def _get_parquet_columns(self):
        parquet_file = pq.ParquetFile(self.file_path)
        metadata: dict = self.get_parquet_metadata()
        cols = [col for col in parquet_file.schema.names if col not in metadata['index_columns']]
        return cols

    def _get_parquet_length(self):
        parquet_file = pq.ParquetFile(self.file_path)
        return parquet_file.metadata.num_rows

    def get_parquet_metadata(self) -> dict:
        parquet_file = pq.ParquetFile(self.file_path)
        pd_metadata_bytes = parquet_file.metadata.metadata.get(b'pandas')
        pd_metadata_str: str = pd_metadata_bytes.decode('utf-8')
        return json.loads(pd_metadata_str)

    def get_index(self) -> pd.Index:
        parquet_file = pq.ParquetFile(self.file_path)
        pd_metadata: dict = self.get_parquet_metadata()
        index_columns = pd_metadata['index_columns']
        # deal with the single range index case
        if len(index_columns) == 1:
            if index_columns[0].get('kind') == 'range':
                df_index = pd.Index(
                    range(index_columns[0].get('start'), index_columns[0].get('stop'), index_columns[0].get('step')))
            else:
                df_index = pd.Index(parquet_file.read(columns=index_columns[0].get('name')).to_pandas())
        else:
            # extract the pd.MultiIndex
            df_index = parquet_file.read(columns=index_columns).to_pandas().index
        return df_index

    def read(self, columns: Optional[list[str]] = None, query: Optional[str] = None,
             with_index: bool = True) -> pd.DataFrame:
        # If no columns are specified, load all columns
        if not columns:
            columns = self.variables_in_file
        else:
            # Check if the columns specified are valid
            for col in columns:
                if col not in self.variables_in_file:
                    raise ValueError(f"Column '{col}' not found in the Parquet file: {self.file_path}.  "
                                     f"Available columns are: {self.variables_in_file}")

        # If a query is specified, parse it to find the columns involved
        if query:
            query_columns = self._parse_query_columns(query)
            # Load only the columns involved in the query
            parquet_file = pq.ParquetFile(self.file_path)
            df_query = parquet_file.read(columns=query_columns).to_pandas()  # Apply the query to the DataFrame
            df_query = df_query.query(query)
            # Get the indices of the rows that match the query
            query_indices = df_query.index
            # Load the remaining columns, but only for the rows that match the query
            remaining_columns = [col for col in columns if col not in query_columns]
            if remaining_columns:
                chunks = []
                for col in remaining_columns:
                    df_col = parquet_file.read(columns=[col]).to_pandas()
                    chunks.append(df_col.loc[query_indices])
                # Concatenate the query DataFrame and the remaining DataFrame
                df = pd.concat([df_query, *chunks], axis=1)
            else:
                df = df_query
            if with_index:
                df_index: pd.Index = self.get_index()[query_indices]
                df.set_index(df_index, inplace=True, drop=True)

        else:
            # If no query is specified, load the specified columns
            df = pd.read_parquet(self.file_path, columns=columns)
            if with_index is False:
                df.reset_index(drop=True, inplace=True)

        return df


class OMFFileReader(BaseReader):
    """
    Read an OMF file
    """

    def __init__(self, file_path, element: str):
        """
        Initialize the OMF file reader.  The element must be a VolumeElement in the OMF file.
        Args:
            file_path: The path to the OMF file
            element: The name of the element in the OMF file to be validated. E.g. 'Block Model'
        """
        super().__init__(file_path)

        # check that the element provided is a valid VolumeElement in the OMF file.
        self.elements = OMFReader(str(file_path)).get_project().elements
        self.element_names = [e.name for e in self.elements]
        if element not in self.element_names:
            raise ValueError(f"Element '{element}' not found in the OMF file: {file_path}. Available elements are:"
                             f" {list(self.elements.keys())}")
        elif self.get_element_by_name(element).__class__.__name__ != 'VolumeElement':
            raise ValueError(f"Element '{element}' is not a VolumeElement in the OMF file: {file_path}")

        self.element = self.get_element_by_name(element)

        self.variables_in_file = [v.name for v in self.element.data]
        self.records_in_file = len(self.element.data[0].array.array)

    def get_element_by_name(self, element_name: str):
        # get the index of the element in order to index into elements
        element_index = self.element_names.index(element_name)
        return self.elements[element_index]

    def read(self, columns: Optional[list[str]] = None, query: Optional[str] = None,
             with_index: bool = True) -> pd.DataFrame:
        # Get the VolumeElement from the OMF file
        # volume_element = OMFReader(self.file_path).get_project().elements[self.element]

        # If no columns are specified, load all columns
        if not columns:
            columns = self.variables_in_file
        else:
            # Check if the columns specified are valid
            for col in columns:
                if col not in self.variables_in_file:
                    raise ValueError(f"Column '{col}' not found in the VolumeElement: {self.element}")

        # If a query is specified, parse it to find the columns involved
        if query:
            query_columns = self._parse_query_columns(query)
            # Load only the columns involved in the query
            df_query: pd.DataFrame = self.read_volume_variables(self.element, variables=query_columns)
            # Apply the query to the DataFrame
            df_query = df_query.query(query)
            # Get the indices of the rows that match the query
            query_indices = df_query.index
            # Load the remaining columns, but only for the rows that match the query
            remaining_columns = [col for col in columns if col not in query_columns]
            if remaining_columns:
                chunks = []
                for col in remaining_columns:
                    data_array = self.read_volume_variables(self.element, variables=[col])
                    # Filter the numpy array using the query indices
                    filtered_data_array = data_array[query_indices]
                    # Convert the filtered numpy array to a DataFrame
                    chunks.append(pd.DataFrame(filtered_data_array, columns=[col]))
                # Concatenate the query DataFrame and the remaining DataFrame
                df = pd.concat([df_query, *chunks], axis=1)
            else:
                df = df_query
        else:
            # If no query is specified, load the specified columns
            df = self.read_volume_variables(self.element, variables=columns)

        # add the index
        if with_index:
            df.set_index(self.get_index(), inplace=True, drop=True)

        return df

    def get_index(self) -> pd.MultiIndex:

        geometry: VolumeGridGeometry = self.element.geometry
        ox, oy, oz = geometry.origin

        # Make coordinates (points) along each axis, i, j, k
        i = ox + np.cumsum(geometry.tensor_u)
        i = np.insert(i, 0, ox)
        j = oy + np.cumsum(self.element.geometry.tensor_v)
        j = np.insert(j, 0, oy)
        k = oz + np.cumsum(self.element.geometry.tensor_w)
        k = np.insert(k, 0, oz)

        # convert to centroids
        x, y, z = (i[1:] + i[:-1]) / 2, (j[1:] + j[:-1]) / 2, (k[1:] + k[:-1]) / 2
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Calculate dx, dy, dz
        dxx, dyy, dzz = np.meshgrid(geometry.tensor_u, geometry.tensor_v, geometry.tensor_w, indexing="ij")

        # TODO: consider rotation

        index = pd.MultiIndex.from_arrays([xx.ravel("F"), yy.ravel("F"), zz.ravel("F"),
                                           dxx.ravel("F"), dyy.ravel("F"), dzz.ravel("F")],
                                          names=['x', 'y', 'z', 'dx', 'dy', 'dz'])

        if len(index) != self.records_in_file:
            raise ValueError(f"The length of the index ({len(index)}) does not match the number of records"
                             f" in the VolumeElement ({self.records_in_file})")

        return index

    def read_volume_variables(self, element: str, variables: list[str]) -> pd.DataFrame:
        # Loop over the variables
        chunks: list[pd.DataFrame] = []
        for variable in variables:
            # Check if the variable exists in the VolumeElement
            if variable not in self.variables_in_file:
                raise ValueError(f"Variable '{variable}' not found in the VolumeElement: {element}")
            chunks.append(self._get_variable_by_name(variable).ravel())

        # Concatenate all chunks into a single DataFrame
        return pd.DataFrame(np.vstack(chunks), index=variables).T

    def _get_variable_by_name(self, variable_name: str):
        # get the index of the variable in order to index into elements
        variable_index = self.variables_in_file.index(variable_name)
        return self.element.data[variable_index].array.array


class ParquetFileWriter:

    def __init__(self):
        pass

    @classmethod
    def from_column_generator(cls, index: pd.Index, column_generator):

        # Path to the final output file
        output_file = "final.parquet"

        # Temp directory for storing parquet columns
        temp_dir = "temp/"

        # Ensure the temp directory exists
        os.makedirs(temp_dir, exist_ok=True)

        # Write the index to a separate Parquet file
        index_table = pa.Table.from_pandas(index.to_frame('index'))
        pq.write_table(index_table, temp_dir + "index.parquet")
        index_pf = pq.ParquetFile(temp_dir + "index.parquet")

        for i, column in enumerate(column_generator):
            # Write each column to a temporary parquet file
            table = pa.Table.from_pandas(column.to_frame())
            pq.write_table(table, temp_dir + f"column_{i}.parquet")

        # Collect paths to the temporary Parquet files
        paths = [temp_dir + file for file in os.listdir(temp_dir) if file != "index.parquet"]

        # Create a ParquetWriter for the final output file
        first_pf = pq.ParquetFile(paths[0])
        writer = pq.ParquetWriter(output_file, first_pf.schema)

        for i in range(index_pf.num_row_groups):
            # Read index chunk
            index_chunk = index_pf.read_row_group(i).to_pandas()

            # Dataframe to store chunk data
            df = pd.DataFrame(index=index_chunk['index'])

            for path in paths:
                pf = pq.ParquetFile(path)
                # Read data chunk
                data_chunk = pf.read_row_group(i).to_pandas()

                # Concatenate data chunk to the dataframe
                df = pd.concat([df, data_chunk], axis=1)

            # Write the chunk to the output file
            writer.write_table(pa.Table.from_pandas(df))

        # Close the writer and release resources
        writer.close()

        # Remove temporary files
        for file in os.listdir(temp_dir):
            os.remove(temp_dir + file)
