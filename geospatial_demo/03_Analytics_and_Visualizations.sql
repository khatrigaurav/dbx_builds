-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Analytics and Visualizations (Gold Layer)
-- MAGIC
-- MAGIC This notebook:
-- MAGIC 1. Creates Gold layer aggregations (state & county)
-- MAGIC 2. Generates key visualizations
-- MAGIC 3. Provides dashboard queries

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Part 1: Gold Layer Aggregations

-- COMMAND ----------

USE CATALOG geospatial_demo;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 1. Earthquakes by State

-- COMMAND ----------

CREATE OR REPLACE TABLE geospatial_demo.gold.earthquakes_by_state AS
SELECT 
    s.state_fips,
    s.state_code,
    s.state_name,
    s.geometry,
    s.centroid_lat,
    s.centroid_lon,
    
    -- Simple metrics
    COUNT(e.event_id) as earthquake_count,
    ROUND(AVG(e.magnitude), 2) as avg_magnitude,
    ROUND(MAX(e.magnitude), 2) as max_magnitude,
    COUNT(CASE WHEN e.magnitude >= 5.0 THEN 1 END) as significant_count
    
FROM geospatial_demo.silver.us_states s
LEFT JOIN geospatial_demo.silver.earthquakes e
    ON ST_CONTAINS(try_to_geometry(s.geometry), try_to_geometry(e.location_point))
GROUP BY s.state_fips,s.state_code, s.state_name, s.geometry, s.centroid_lat, s.centroid_lon;

SELECT * FROM geospatial_demo.gold.earthquakes_by_state 
WHERE earthquake_count > 0 
ORDER BY earthquake_count DESC 
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 2. Earthquakes by County

-- COMMAND ----------

CREATE OR REPLACE TABLE geospatial_demo.gold.earthquakes_by_county AS
SELECT 
    c.state_fips,
    c.county_name,
    c.geometry,
    c.centroid_lat,
    c.centroid_lon,
    s.state_name,
    s.state_code,
    
    -- Simple metrics
    COUNT(e.event_id) as earthquake_count,
    ROUND(AVG(e.magnitude), 2) as avg_magnitude,
    ROUND(MAX(e.magnitude), 2) as max_magnitude
    
FROM geospatial_demo.silver.us_counties c
LEFT JOIN geospatial_demo.silver.us_states s
    ON c.state_fips = s.state_fips 
LEFT JOIN geospatial_demo.silver.earthquakes e
    ON ST_CONTAINS(try_to_geometry(c.geometry), try_to_geometry(e.location_point))
GROUP BY c.state_fips, c.county_name, c.geometry, c.centroid_lat, c.centroid_lon, s.state_name, s.state_code;

SELECT * FROM geospatial_demo.gold.earthquakes_by_county 
WHERE earthquake_count > 0 
ORDER BY earthquake_count DESC 
-- LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Part 2: Visualizations

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC import plotly.express as px
-- MAGIC import plotly.graph_objects as go
-- MAGIC # from shapely import wkt

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 1. Interactive Point Map - Earthquake Locations

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Load earthquake data
-- MAGIC df_eq = spark.table("geospatial_demo.silver.earthquakes").toPandas()
-- MAGIC
-- MAGIC # Filter for visualization
-- MAGIC df_eq_viz = df_eq[df_eq['magnitude'] >= 3.0].copy()
-- MAGIC
-- MAGIC # Create interactive map
-- MAGIC fig = px.scatter_geo(
-- MAGIC     df_eq_viz,
-- MAGIC     lat='latitude',
-- MAGIC     lon='longitude',
-- MAGIC     size='magnitude',
-- MAGIC     color='magnitude',
-- MAGIC     hover_name='place',
-- MAGIC     hover_data={'magnitude': ':.2f', 'depth_km': ':.1f', 'event_timestamp': True},
-- MAGIC     title='Earthquake Locations (Magnitude ≥ 3.0)',
-- MAGIC     color_continuous_scale='Reds',
-- MAGIC     size_max=15
-- MAGIC )
-- MAGIC
-- MAGIC fig.update_geos(scope='usa', projection_type='albers usa')
-- MAGIC fig.update_layout(height=600)
-- MAGIC fig.show()

-- COMMAND ----------

SELECT * FROM geospatial_demo.gold.earthquakes_by_state

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 2. State-Level Choropleth Map

-- COMMAND ----------

SELECT * FROM geospatial_demo.gold.earthquakes_by_state 
WHERE earthquake_count > 0 
ORDER BY earthquake_count DESC 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 3. Time Series Analysis

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Daily earthquake counts
-- MAGIC df_daily = spark.sql("""
-- MAGIC     SELECT 
-- MAGIC         event_date,
-- MAGIC         COUNT(*) as earthquake_count,
-- MAGIC         ROUND(AVG(magnitude), 2) as avg_magnitude,
-- MAGIC         ROUND(MAX(magnitude), 2) as max_magnitude
-- MAGIC     FROM geospatial_demo.silver.earthquakes
-- MAGIC     GROUP BY event_date
-- MAGIC     ORDER BY event_date
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC # Create time series plot
-- MAGIC fig = go.Figure()
-- MAGIC
-- MAGIC fig.add_trace(go.Scatter(
-- MAGIC     x=df_daily['event_date'],
-- MAGIC     y=df_daily['earthquake_count'],
-- MAGIC     mode='lines+markers',
-- MAGIC     name='Daily Count',
-- MAGIC     line=dict(color='steelblue', width=2)
-- MAGIC ))
-- MAGIC
-- MAGIC fig.update_layout(
-- MAGIC     title='Daily Earthquake Frequency',
-- MAGIC     xaxis_title='Date',
-- MAGIC     yaxis_title='Number of Earthquakes',
-- MAGIC     height=500
-- MAGIC )
-- MAGIC fig.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 4. Magnitude Distribution

-- COMMAND ----------

-- MAGIC %python
-- MAGIC fig = px.histogram(
-- MAGIC     df_eq,
-- MAGIC     x='magnitude',
-- MAGIC     nbins=40,
-- MAGIC     title='Earthquake Magnitude Distribution',
-- MAGIC     labels={'magnitude': 'Magnitude', 'count': 'Count'},
-- MAGIC     color_discrete_sequence=['steelblue']
-- MAGIC )
-- MAGIC fig.update_layout(height=500)
-- MAGIC fig.show()