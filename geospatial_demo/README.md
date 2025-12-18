# Databricks Geospatial Demo - Simplified Version

This  demo showcases Databricks geospatial capabilities using real-time earthquake data from the USGS Earthquake Catalog. 
It demonstrates the power of native GEOGRAPHY and GEOMETRY data types in Databricks, combined with a robust medallion architecture that powers comprehensive visualizations in notebooks/ dashboards


## 🎯 What This Demo Shows

- **GEOGRAPHY & GEOMETRY data types** in Databricks
- **Spatial operations**: ST_POINT, ST_CONTAINS, ST_DISTANCE
- **Spatial joins**: Point-in-polygon operations
- **Choropleth maps**: State and county visualizations
- **Medallion architecture**: Bronze → Silver → Gold

## 📁 Files 

```
Project Folder/
├── 01_Setup_and_Ingestion.py          # Creates schemas + ingests data
├── 02_Geospatial_Transformations.py   # Adds GEOGRAPHY/GEOMETRY types
├── 03_Analytics_and_Visualizations.sql # Aggregations + viz + queries
├── Geospatial Dashboard                  # Dashboard that feeds on the gold layer
├── simple_config.json                  # Configuration
└── README.md                          # This file
```

## 🚀 Quick Start (20 minutes)

### Prerequisites
- Databricks workspace 
- Unity Catalog enabled
- Internet access (for API calls)

### Steps

**1. Update Configuration** (1 minute)

Edit `simple_config.json` if needed:
```json
{
  "databricks": {
    "catalog": "geospatial_demo"
  }
}
```

Update notebook paths from `/Workspace/Repos/your_username/GEO_II/` to your actual path.

**2. Run Notebooks** (20 minutes)

```bash
# Step 1: Setup and ingest data (10 min)
01_Setup_and_Ingestion.py

# Step 2: Add geospatial types (5 min)
02_Geospatial_Transformations.py

# Step 3: Create analytics and visualizations (5 min)
03_Analytics_and_Visualizations.sql

# Step 4: Run Dashboard
Geogspatial Dashboard
```

That's it! You now have:
- ✅ Earthquake data with GEOGRAPHY/GEOMETRY types
- ✅ State and county spatial aggregations
- ✅ Interactive visualizations
- ✅ Dashboard-ready queries

## 📊 What Gets Created

### Data Tables

**Bronze Layer** (Raw data):
- `earthquakes_raw` - 7 columns: id, time, lat, lon, depth, magnitude, place
- `us_states_raw` - 3 columns: state_code, state_name, geometry_wkt
- `us_counties_raw` - 3 columns: state_fips, county_name, geometry_wkt

**Silver Layer** (Geospatial):
- `earthquakes` - 10 columns including **location_point** and **location_geography** 
- `us_states` - 6 columns including **geometry** and centroids
- `us_counties` - 5 columns including **geometry** and centroids

**Gold Layer** (Analytics):
- `earthquakes_by_state` - 9 columns: state info + earthquake metrics
- `earthquakes_by_county` - 10 columns: county info + earthquake metrics

### Visualizations

1. **Point Map** - Earthquake locations (magnitude ≥ 3.0)
2. **Bar Chart** - Top 15 states by earthquake count
3. **Time Series** - Daily earthquake frequency
4. **Histogram** - Magnitude distribution

## 🎓 Key Concepts Demonstrated

### 1. Geospatial Data Types
- **GEOGRAPHY** - Earth-aware coordinates with accurate distance calculations
- **GEOMETRY** - 2D spatial objects for mapping and visualization
- Conversion between WKT, WKB, and native types

```python
# Convert coordinates to GEOMETRY
.withColumn('location_point', expr('ST_POINT(longitude, latitude)'))

# Convert to GEOGRAPHY (Earth-aware)
.withColumn('location_geography', 
    expr("ST_GEOGFROMTEXT('POINT(' || longitude || ' ' || latitude || ')')"))
```

### 2. Spatial Joins

```sql
-- Find which state contains each earthquake
SELECT s.state_name, COUNT(e.event_id) as earthquake_count
FROM silver.us_states s
JOIN silver.earthquakes e
    ON ST_CONTAINS(s.geometry, e.location_point)  -- Point-in-polygon!
GROUP BY s.state_name
```
Common Spatial Operations:
- `ST_POINT()` - Create point geometries from coordinates
- `ST_CONTAINS()` - Point-in-polygon spatial joins
- `ST_DISTANCE()` - Distance calculations between geographic points
- `ST_AREA()` - Calculate area of polygons
- `ST_BUFFER()` - Create buffer zones around points
- `ST_CENTROID()` - Find geometric centers
- `ST_WITHIN()`, `ST_INTERSECTS()` - Spatial predicates

### 3. Distance Calculations

```sql
-- Distance between two cities
SELECT ST_DISTANCE(
    ST_GEOGFROMTEXT('POINT(-122.4194 37.7749)'),  -- San Francisco
    ST_GEOGFROMTEXT('POINT(-118.2437 34.0522)')   -- Los Angeles
) / 1000 as distance_km
```

## 📈 Data Sources

- **USGS Earthquake Catalog**: Real-time earthquake data (last 30 days, magnitude ≥ 2.5)
- **US Census TIGER/Line**: State and county boundaries (2021)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  • USGS Earthquake Catalog API (Real-time seismic data)         │
│  • US Census Bureau TIGER/Line (State/County boundaries)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER (Raw Data)                       │
├─────────────────────────────────────────────────────────────────┤
│  • earthquakes_raw (lat/lon, magnitude, depth, metadata)        │
│  • us_states_raw (geometry as WKT)                              │
│  • us_counties_raw (geometry as WKT)                            │
│  Storage: Delta Lake | Format: Raw/Unprocessed                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SILVER LAYER (Enriched & Validated)                 │
├─────────────────────────────────────────────────────────────────┤
│  • earthquakes (GEOGRAPHY/GEOMETRY points, classifications)     │
│  • us_states (GEOMETRY polygons, centroids, area calculations)  │
│  • us_counties (GEOMETRY polygons, detailed boundaries)         │
│  Storage: Delta Lake | Format: Enriched + Geospatial Types      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           GOLD LAYER (Aggregated Analytics)                      │
├─────────────────────────────────────────────────────────────────┤
│  • earthquakes_by_state (Spatial aggregations, risk scores)     │
│  • earthquakes_by_county (Detailed regional metrics)            │
│  Storage: Delta Lake | Format: Business-Ready Analytics         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONSUMPTION LAYER                                   │
├─────────────────────────────────────────────────────────────────┤
│  • Databricks SQL Dashboards (Interactive visualizations)       │
│  • Python/R Notebooks (Advanced analytics)                      │
└─────────────────────────────────────────────────────────────────┘
```


**KPIs:**
- Total earthquakes
- Average magnitude
- Max magnitude event

**Charts:**
- Top 10 states (table/bar chart)
- Recent earthquakes (table)
- Magnitude distribution (bar chart)
- State map (choropleth)
- County map (choropleth)

## 🔧 Column Reference

### Silver Layer - earthquakes (10 columns)
| Column | Type | Description |
|--------|------|-------------|
| event_id | string | Unique identifier |
| event_timestamp | timestamp | When earthquake occurred |
| event_date | date | Date only (for partitioning) |
| latitude | double | Latitude coordinate |
| longitude | double | Longitude coordinate |
| depth_km | double | Depth in kilometers |
| magnitude | double | Earthquake magnitude |
| place | string | Location description |
| location_point | string | Spatial point |
| location_geography | string | Earth-aware point |

### Silver Layer - us_states (5 columns)
| Column | Type | Description |
|--------|------|-------------|
| state_fips | string | Two-digit state FIPS code |
| state_code | string | Two-letter state code |
| state_name | string | Full state name |
| geometry | string| State boundary polygon |
| centroid_lat | double | Center latitude |
| centroid_lon | double | Center longitude |

### Silver Layer - us_counties (5 columns)
| Column | Type | Description |
|--------|------|-------------|
| state_fips | string | Two-digit state FIPS code |
| county_name | string | Full state name |
| geometry | string | County boundary polygon |
| centroid_lat | double | Center latitude |
| centroid_lon | double | Center longitude |

### Gold Layer - earthquakes_by_state (9 columns)
| Column | Type | Description |
|--------|------|-------------|
| state_code | string | Two-letter state code |
| state_name | string | Full state name |
| geometry | GEOMETRY | State boundary |
| centroid_lat | double | Center latitude |
| centroid_lon | double | Center longitude |
| earthquake_count | long | Number of earthquakes |
| avg_magnitude | double | Average magnitude |
| max_magnitude | double | Maximum magnitude |
| significant_count | long | Count of M≥5.0 events |

### Gold Layer - earthquakes_by_county
| Column | Type | Description |
|--------|------|-------------|
| state_fips | string | Two-digit state FIPS code |
| county_name | string | Full state name |
| geometry | GEOMETRY | County boundary |
| centroid_lat | double | Center latitude |
| centroid_lon | double | Center longitude |
| state_code | string | Two-letter state code |
| state_name | string | Full state name |
| earthquake_count | long | Number of earthquakes |
| avg_magnitude | double | Average magnitude |
| max_magnitude | double | Maximum magnitude |

## 🎯 Use Cases

This simplified demo shows how to:
- Ingest geospatial data from APIs
- Convert coordinates to native spatial types
- Perform spatial joins (point-in-polygon)
- Calculate distances and areas
- Create choropleth maps
- Build production-ready analytics tables

## ⚡ Performance

- Bronze ingestion: ~10 minutes
- Silver transformation: ~5 minutes
- Gold aggregations: ~5 minutes
- **Total: ~20 minutes**

## 🛠️ Troubleshooting

**Issue: "Catalog not found"**
→ Run notebook 01 first to create schemas

**Issue: "ST_POINT not found"**
→ Ensure DBR version is 13.0+

**Issue: "No earthquakes found"**
→ Check API connectivity and date range

## 📚 Next Steps

1. **Customize date range** in `simple_config.json`
2. **Add more visualizations** in notebook 03
3. **Create Databricks SQL Dashboard** using provided queries
4. **Extend with ML models** for earthquake prediction
5. **Add real-time streaming** for live monitoring

## 📖 Learn More

- [Databricks Geospatial Functions](https://docs.databricks.com/sql/language-manual/sql-ref-functions-builtin.html#geospatial-functions)
- [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)
- [Delta Lake Guide](https://docs.databricks.com/delta/index.html)

---

Run the 3 notebooks in order and you'll have a complete geospatial analytics pipeline.

