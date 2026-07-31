import ee
ee.Initialize(project="msc-project-504108")

chirps_monthly = ee.ImageCollection("UCSB-CHG/CHIRPS/MONTHLY").filterDate(
    "2018-01-01", "2024-12-31"
)

precipitation = chirps_monthly.select('precipitation')