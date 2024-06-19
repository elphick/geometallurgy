"""
Create Block Model
==================

We leverage the omfvista block model example.  We load the model and convert to a parquet.

Later, we may use this model along with a correlation matrix for an iron ore dataset to create a pseudo-realistic
iron ore block model for testing.

We can also up-sample the grid to create larger datasets for testing.

# REF: https://opengeovis.github.io/omfvista/examples/load-project.html#sphx-glr-examples-load-project-py

"""

import omfvista
import pooch
import pyvista as pv
import pandas as pd
from omf import VolumeElement
from ydata_profiling import ProfileReport

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

# Now you can load the file using omfvista
project = omfvista.load_project(file_path)
print(project)

# %%
project.plot()

# %%

vol = project["Block Model"]
assay = project["wolfpass_WP_assay"]
topo = project["Topography"]
dacite = project["Dacite"]

assay.set_active_scalars("DENSITY")

p = pv.Plotter()
p.add_mesh(assay.tube(radius=3))
p.add_mesh(topo, opacity=0.5)
p.show()

# %%
# Threshold the volumetric data
thresh_vol = vol.threshold([1.09, 4.20])
print(thresh_vol)

# %%
# Create a plotting window
p = pv.Plotter()
# Add the bounds axis
p.show_bounds()
p.add_bounding_box()

# Add our datasets
p.add_mesh(topo, opacity=0.5)
p.add_mesh(
    dacite,
    color="orange",
    opacity=0.6,
)
# p.add_mesh(thresh_vol, cmap="coolwarm", clim=vol.get_data_range())
p.add_mesh_threshold(vol, scalars="CU_pct", show_edges=True)


# Add the assay logs: use a tube filter that varius the radius by an attribute
p.add_mesh(assay.tube(radius=3), cmap="viridis")

p.show()

# %%
# Export the model data
# ---------------------

# Create DataFrame
df = pd.DataFrame(vol.cell_centers().points, columns=['x', 'y', 'z'])

# Add the array data to the DataFrame
for name in vol.array_names:
    df[name] = vol.get_array(name)

# set the index to the cell centroids
df.set_index(['x', 'y', 'z'], drop=True, inplace=True)

# Write DataFrame to parquet file
df.to_parquet('block_model_copper.parquet')

# %%
# Profile
# -------

profile = ProfileReport(df.reset_index(), title="Profiling Report")
profile.to_file("block_model_copper_profile.html")
