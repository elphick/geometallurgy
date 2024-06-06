"""
Consuming OMF
=============

This example demonstrates how to consume and Open Mining Format file
"""
import omf
import pooch
import json

# %%
# Load
# ----

# Base URL and relative path
base_url = "https://github.com/OpenGeoVis/omfvista/raw/master/assets/"
relative_path = "test_file.omf"

# Create a Pooch object
p = pooch.create(
    path=pooch.os_cache("geometallurgy"),
    base_url=base_url,
    registry={relative_path: None}
)

# Use fetch method to download the file
file_path = p.fetch(relative_path)

reader = omf.OMFReader(file_path)
project: omf.Project = reader.get_project()
print(project.name)
print(project.elements)
print(project.description)
