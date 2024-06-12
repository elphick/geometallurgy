from typing import Optional, Union

import pandas as pd
import pandera as pa
import yaml

df = pd.DataFrame({
    "column1": [5, 10, 20],
    "column2": ["a", "b", "c"],
    "column3": pd.to_datetime(["2010", "2011", "2012"]),
})
schema = pa.infer_schema(df)
print(schema)

# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
yaml_schema = schema.to_yaml()
print(yaml_schema)


# create a function that creates they yml format of a column schema manually
def create_column_schema(column_name: str,
                         data_type: str, nullable: bool = True,
                         title: Optional[str] = None,
                         description: Optional[str] = None,
                         value_range: Optional[list] = None,
                         unique: Optional[bool] = False,
                         coerce: Optional[bool] = False,
                         required: Optional[bool] = True,
                         regex: Optional[Union[str, bool]] = False) -> dict:
    d_schema: dict = {
        column_name: {"title": title, "description": description, "dtype": data_type, "nullable": nullable}}
    if value_range:
        d_schema[column_name]['checks'] = {"greater_than_or_equal_to": value_range[0],
                                           "less_than_or_equal_to": value_range[1]}
    d_schema[column_name]['unique'] = unique
    d_schema[column_name]['coerce'] = coerce
    d_schema[column_name]['required'] = required
    d_schema[column_name]['regex'] = regex
    return d_schema


str_schema: dict = create_column_schema(column_name='my_column', data_type='int64', nullable=False)
yaml_data = yaml.dump(str_schema, sort_keys=False)

print(yaml_data)

print('done')
