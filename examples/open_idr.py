import napari

uri = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001241.zarr"

viewer = napari.Viewer()
viewer.open(uri, plugin="napari-ome-zarr")

napari.run()


