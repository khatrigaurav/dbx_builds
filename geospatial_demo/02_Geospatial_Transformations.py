# Databricks notebook source
# MAGIC %md
# MAGIC # Geospatial Transformations (Silver Layer)
# MAGIC
# MAGIC This notebook converts raw data to geospatial types with only essential columns.

# COMMAND ----------

import json
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Load configuration
with open('simple_config.json', 'r') as f:
    config = json.load(f)

catalog = config['databricks']['catalog']
bronze_schema = config['databricks']['bronze_schema']
silver_schema = config['databricks']['silver_schema']

spark.sql(f"USE CATALOG {catalog}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Transform Earthquakes - Add GEOGRAPHY/GEOMETRY Types

# COMMAND ----------

# Read Bronze data
df_earthquakes_raw = spark.table(f"{catalog}.{bronze_schema}.earthquakes_raw")

# Clean and add geospatial types - ESSENTIAL COLUMNS ONLY
df_earthquakes_silver = df_earthquakes_raw \
    .filter(col('event_id').isNotNull()) \
    .filter(col('latitude').isNotNull() & col('longitude').isNotNull()) \
    .filter((col('latitude') >= -90) & (col('latitude') <= 90)) \
    .filter((col('longitude') >= -180) & (col('longitude') <= 180)) \
    .dropDuplicates(['event_id']) \
    .withColumn('event_timestamp', (col('event_time') / 1000).cast('timestamp')) \
    .withColumn('event_date', to_date((col('event_time') / 1000).cast('timestamp'))) \
    .withColumn('location_point', expr('st_aswkt(ST_POINT(longitude, latitude))')) \
    .withColumn('location_geography', expr("st_aswkt(ST_GEOGFROMTEXT('POINT(' || longitude || ' ' || latitude || ')'))")) \
    .select(
        'event_id',
        'event_timestamp',
        'event_date',
        'latitude',
        'longitude',
        'depth_km',
        'magnitude',
        'place',
        'location_point',
        'location_geography'
    )

# Write to Silver
df_earthquakes_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{silver_schema}.earthquakes")

print(f"✓ Silver earthquakes: {df_earthquakes_silver.count()} records with 10 columns")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from geospatial_demo.silver.earthquakes limit 4

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform States - Add GEOMETRY Types

# COMMAND ----------

# Read Bronze states (need to add state_fips from raw data)
df_states_raw = spark.table(f"{catalog}.{bronze_schema}.us_states_raw")

# Check if state_fips exists in raw data, if not we'll create a mapping
# For now, let's add it from the original Bronze table
df_states_bronze_full = spark.sql(f"""
    SELECT state_code,  state_name, state_fips, geometry_wkt
    FROM {catalog}.{bronze_schema}.us_states_raw
""")


# Read and transform states
df_states = df_states_bronze_full \
    .withColumn('geometry_', expr('(ST_GEOMFROMTEXT(geometry_wkt))')) \
    .withColumn('centroid_lat', expr('ST_Y(ST_CENTROID(geometry_))')) \
    .withColumn('centroid_lon', expr('ST_X(ST_CENTROID(geometry_))')) \
    .withColumn('geometry', expr('ST_ASWKT(ST_GEOMFROMTEXT(geometry_wkt))')) \
    .select(
        'state_fips',
        'state_code',
        'state_name',
        'geometry',
        'centroid_lat',
        'centroid_lon'
    )

df_states.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{silver_schema}.us_states")

print(f"✓ Silver states: {df_states.count()} records with 5 columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Transform Counties - Add GEOMETRY Types

# COMMAND ----------

# Read and transform counties
df_counties = spark.table(f"{catalog}.{bronze_schema}.us_counties_raw") \
    .withColumn('geometry_', expr('ST_GEOMFROMTEXT(geometry_wkt)')) \
    .withColumn('centroid_lat', expr('ST_Y(ST_CENTROID(geometry_))')) \
    .withColumn('centroid_lon', expr('ST_X(ST_CENTROID(geometry_))')) \
    .withColumn('geometry', expr('ST_ASWKT(ST_GEOMFROMTEXT(geometry_wkt))')) \
    .select(
        'state_fips',
        'county_name',
        'geometry',
        'centroid_lat',
        'centroid_lon'
    )

df_counties.write.format("delta").mode("overwrite").partitionBy("state_fips").saveAsTable(f"{catalog}.{silver_schema}.us_counties")

print(f"✓ Silver counties: {df_counties.count()} records with 5 columns")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from  geospatial_demo.bronze.us_states_raw limit 4;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Spatial Operations

# COMMAND ----------

# Test spatial join
spark.sql(f"""
    SELECT 
        s.state_name,
        COUNT(e.event_id) as earthquake_count,
        ROUND(AVG(e.magnitude), 2) as avg_magnitude,
        ROUND(MAX(e.magnitude), 2) as max_magnitude
    FROM {catalog}.{silver_schema}.us_states s
    LEFT JOIN {catalog}.{silver_schema}.earthquakes e
        ON ST_CONTAINS(st_geomfromwkt(s.geometry), st_geomfromwkt(e.location_point))
    GROUP BY s.state_name
    ORDER BY earthquake_count DESC
    LIMIT 10
""").show()

# COMMAND ----------

# Test distance calculation
spark.sql(f"""
    SELECT 
        ST_DISTANCE(
            ST_GEOMFROMTEXT('POINT(-122.4194 37.7749)'),  -- San Francisco
            ST_GEOMFROMTEXT('POINT(-118.2437 34.0522)')   -- Los Angeles
        ) / 1000 as distance_km
""").show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Silver layer created with minimal essential columns:
# MAGIC
# MAGIC **Earthquakes** (10 columns):
# MAGIC - event_id, event_timestamp, event_date
# MAGIC - latitude, longitude, depth_km, magnitude, place
# MAGIC - location_point (GEOMETRY), location_geography (GEOGRAPHY)
# MAGIC
# MAGIC **States** (5 columns):
# MAGIC - state_code, state_name, geometry, centroid_lat, centroid_lon
# MAGIC
# MAGIC **Counties** (5 columns):
# MAGIC - state_fips, county_name, geometry, centroid_lat, centroid_lon
# MAGIC
# MAGIC Next: Run `03_Analytics_and_Visualizations.sql`
# MAGIC