# Databricks notebook source
# MAGIC %md
# MAGIC # Setup and Data Ingestion
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates Unity Catalog schemas
# MAGIC 2. Installs geospatial libraries
# MAGIC 3. Ingests earthquake data from USGS API
# MAGIC 4. Ingests state/county boundaries from Census Bureau

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Environment

# COMMAND ----------

# Install required libraries
%pip install geopandas folium plotly requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

import requests
import json
from datetime import datetime, timedelta, timezone
from pyspark.sql.functions import *
from pyspark.sql.types import *
import geopandas as gpd
from io import BytesIO
from zipfile import ZipFile

# Load configuration
config_path = 'simple_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print("✓ Libraries loaded")

# COMMAND ----------

# Create catalog and schemas
catalog = config['databricks']['catalog']
bronze_schema = config['databricks']['bronze_schema']
silver_schema = config['databricks']['silver_schema']
gold_schema = config['databricks']['gold_schema']

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{bronze_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{silver_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{gold_schema}")

spark.sql(f"USE CATALOG {catalog}")
print(f"✓ Created schemas in {catalog}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 Test Geospatial Functions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   typeof((ST_POINT(-122.4194, 37.7749))) as data_type,
# MAGIC   ST_ASTEXT(ST_POINT(-122.4194, 37.7749)) as san_francisco_point,
# MAGIC   ST_DISTANCE(
# MAGIC     ST_POINT(-122.4194, 37.7749),  -- San Francisco
# MAGIC     ST_POINT(-118.2437, 34.0522)   -- Los Angeles
# MAGIC   ) / 1000 as distance_km,
# MAGIC   'San Francisco to Los Angeles' as route
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest Earthquake Data from USGS

# COMMAND ----------

# Fetch earthquake data
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=config['usgs_api']['days_back'])

params = {
    'format': 'geojson',
    'starttime': start_date.strftime('%Y-%m-%d'),
    'endtime': end_date.strftime('%Y-%m-%d'),
    'minmagnitude': config['usgs_api']['min_magnitude']
}

response = requests.get(config['usgs_api']['base_url'], params=params)
earthquake_data = response.json()

print(f"✓ Fetched {earthquake_data['metadata']['count']} earthquakes")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

features = earthquake_data['features']
earthquake_records = []

for feature in features:
    props = feature['properties']
    coords = feature['geometry']['coordinates']
    
    earthquake_records.append({
        'event_id': feature['id'],
        'event_time': props.get('time'),
        'latitude': coords[1],
        'longitude': coords[0],
        'depth_km': coords[2] if len(coords) > 2 else None,
        'magnitude': props.get('mag'),
        'place': props.get('place')
    })


schema = StructType([
    StructField('event_id', StringType(), True),
    StructField('event_time', LongType(), True),
    StructField('latitude', DoubleType(), True),
    StructField('longitude', DoubleType(), True),
    StructField('depth_km', DoubleType(), True),
    StructField('magnitude', DoubleType(), True),
    StructField('place', StringType(), True)
])

df_earthquakes = spark.createDataFrame(
    earthquake_records,
    schema=schema
)

df_earthquakes.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog}.{bronze_schema}.earthquakes_raw")

print(f"✓ Saved {len(earthquake_records)} earthquakes to Bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingest State Boundaries

# COMMAND ----------

# Download state boundaries
response = requests.get(config['census_boundaries']['states_url'])
with ZipFile(BytesIO(response.content)) as zip_file:
    temp_dir = "/tmp/us_states/"
    dbutils.fs.mkdirs(f"{temp_dir}")
    for file_info in zip_file.filelist:
        zip_file.extract(file_info, f"{temp_dir}")
    gdf_states = gpd.read_file(f"{temp_dir}")

# Convert to Spark DataFrame - ESSENTIAL COLUMNS ONLY
gdf_states['geometry_wkt'] = gdf_states['geometry'].apply(lambda geom: geom.wkt)
pdf_states = gdf_states[['STATEFP', 'STUSPS', 'NAME', 'geometry_wkt']].copy()
pdf_states.columns = ['state_fips', 'state_code', 'state_name', 'geometry_wkt']

df_states = spark.createDataFrame(pdf_states)
df_states.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{bronze_schema}.us_states_raw")

print(f"✓ Saved {df_states.count()} states to Bronze")

# COMMAND ----------

pdf_states.display()

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from geospatial_demo.bronze.us_states_raw limit 4

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ingest County Boundaries

# COMMAND ----------

# Download county boundaries
response = requests.get(config['census_boundaries']['counties_url'])
with ZipFile(BytesIO(response.content)) as zip_file:
    temp_dir = "/tmp/us_counties/"
    dbutils.fs.mkdirs(f"{temp_dir}")
    for file_info in zip_file.filelist:
        zip_file.extract(file_info, f"{temp_dir}")
    gdf_counties = gpd.read_file(f"{temp_dir}")

# Convert to Spark DataFrame - ESSENTIAL COLUMNS ONLY
gdf_counties['geometry_wkt'] = gdf_counties['geometry'].apply(lambda geom: geom.wkt)
pdf_counties = gdf_counties[['STATEFP', 'NAME', 'geometry_wkt']].copy()
pdf_counties.columns = ['state_fips', 'county_name', 'geometry_wkt']

df_counties = spark.createDataFrame(pdf_counties)
df_counties.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{bronze_schema}.us_counties_raw")

print(f"✓ Saved {df_counties.count()} counties to Bronze")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from geospatial_demo.bronze.us_counties_raw limit 4

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Bronze layer created with:
# MAGIC - Earthquakes (7 essential columns)
# MAGIC - States (3 essential columns)
# MAGIC - Counties (3 essential columns)
# MAGIC
# MAGIC Next: Run `02_Geospatial_Transformations.py`
# MAGIC