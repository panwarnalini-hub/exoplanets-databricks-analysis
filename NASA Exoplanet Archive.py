# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring Exoplanets: Databricks Free Edition
# MAGIC Nalini Panwar | demo date: 2025-11-11

# COMMAND ----------

# MAGIC %md
# MAGIC > Every dot you’ll see today is a world beyond our Solar System; a planet orbiting another star.
# MAGIC > NASA’s Exoplanet Archive contains thousands of these alien worlds.
# MAGIC > My goal is to find patterns in that chaos and see if the universe obeys the same rules we learned in school.
# MAGIC
# MAGIC We’ll clean the data, visualize it, and even test a bit of Kepler’s physics **all** inside Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Load and Inspect the NASA Exoplanet Dataset
# MAGIC
# MAGIC The dataset comes from the NASA Exoplanet Archive and contains records of thousands of confirmed exoplanets discovered by various missions such as Kepler and TESS.  
# MAGIC Each row represents one planet, along with its physical characteristics and discovery details.

# COMMAND ----------

raw = spark.read.csv('/Volumes/workspace/default/tables/exoplanets.csv', header=True, inferSchema=True, comment='#')
display(raw.limit(10))

# COMMAND ----------

raw.printSchema()
print("Rows:", raw.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Data Cleaning
# MAGIC
# MAGIC The NASA dataset contains thousands of planet entries with fields for radius, orbital period, stellar temperature, and discovery year.  
# MAGIC We remove incomplete rows and rename columns for clarity so that our data is easy to explore.

# COMMAND ----------

from pyspark.sql.functions import col, when

df = raw.dropna(subset=['pl_name','pl_rade','pl_orbper','st_teff','disc_year'])
df = df.withColumnRenamed('pl_name', 'planet_name') \
       .withColumnRenamed('pl_rade', 'planet_radius') \
       .withColumnRenamed('pl_orbper', 'orbital_period') \
       .withColumnRenamed('st_teff', 'star_temp') \
       .withColumnRenamed('disc_year', 'discovery_year')

# Ensure numeric types (coerce if necessary)
df = df.withColumn('planet_radius', col('planet_radius').cast('double')) \
       .withColumn('orbital_period', col('orbital_period').cast('double')) \
       .withColumn('star_temp', col('star_temp').cast('double')) \
       .withColumn('discovery_year', col('discovery_year').cast('int'))

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC The cleaned dataset now includes:
# MAGIC - **planet_radius** in Earth radii  
# MAGIC - **orbital_period** in days  
# MAGIC - **star_temp** in Kelvin  
# MAGIC - **discovery_year** as the year of detection  
# MAGIC
# MAGIC Each row represents a confirmed exoplanet recorded by NASA’s telescopes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Derive a Simple Habitability Score
# MAGIC
# MAGIC To make the data more insightful, we can define a quick, heuristic *habitability score*.  
# MAGIC This score favors planets with:
# MAGIC - Radii close to Earth’s (around 1 Earth radius)  
# MAGIC - Host star temperatures similar to our Sun (4800-6200 K)
# MAGIC
# MAGIC The idea isn’t to predict real habitability; it’s just a quick feature engineering step to identify planets with roughly Earth-like conditions.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, when, abs as Fabs, round

# crude heuristic: closer to Earth radius (1) and moderate star temp (~5000–6000K) = higher score
df = df.withColumn('radius_score', 1.0 / (1.0 + Fabs(col('planet_radius') - 1.0))) \
       .withColumn('temp_score', when((col('star_temp') >= 4800) & (col('star_temp') <= 6200), 1.0).otherwise(0.2)) \
       .withColumn('habitability_score', round((col('radius_score')* 0.6) + (col('temp_score') * 0.4),3))

display(df.select('planet_name','planet_radius','star_temp','habitability_score')
           .orderBy(col('habitability_score').desc())
           .limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Explore the Data with SQL
# MAGIC
# MAGIC Databricks allows seamless switching between Python and SQL.  
# MAGIC Let’s register the cleaned DataFrame as a temporary view and explore discovery patterns; such as which detection methods have found the most planets, and how discoveries have evolved over time.

# COMMAND ----------

df.createOrReplaceTempView("v_exo")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT planet_name, planet_radius,  ROUND(orbital_period, 2) AS orbital_period, star_temp, discovery_year, habitability_score
# MAGIC FROM v_exo
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT discoverymethod, COUNT(*) AS discoveries
# MAGIC FROM v_exo
# MAGIC GROUP BY discoverymethod
# MAGIC ORDER BY discoveries DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Visual Exploration: Where Might Life Exist?
# MAGIC
# MAGIC We start by plotting orbital period against a habitability indicator.  
# MAGIC Color represents planet size, so small blue dots suggest rocky worlds while larger orange dots indicate gas giants.
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

df_filtered = df.where(F.col("orbital_period") < 1000)

df_filtered = df_filtered \
    .withColumn("planet_radius_rounded", F.round(F.col("planet_radius"), 1)) \
    .withColumn("orbital_period_rounded", F.round(F.col("orbital_period"), 2)) \
    .withColumn("habitability_score_rounded", F.round(F.col("habitability_score"), 3))

display(df_filtered.select(
    "orbital_period_rounded",
    "habitability_score_rounded",
    "planet_radius_rounded"
))


# COMMAND ----------

# MAGIC %md
# MAGIC This chart focuses on planets with orbital periods under 1000 days, which roughly represents the habitable zone of a star.
# MAGIC
# MAGIC **Insight:**  
# MAGIC Earth-sized planets cluster around shorter orbital periods and moderate stellar temperatures.  
# MAGIC This is the range where conditions may allow liquid water and possibly life.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: A Quick Machine Learning Test
# MAGIC
# MAGIC We now train a simple linear regression model to test if a planet’s orbital period can be estimated from its size and its star’s temperature.  
# MAGIC Logarithmic transforms help reveal multiplicative relationships common in astrophysics.
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Convert to pandas for sklearn
pdf = df.select("planet_radius", "star_temp", "orbital_period").dropna().toPandas()

# Apply log transform (helps capture multiplicative relationships)
pdf["log_period"] = np.log10(pdf["orbital_period"])
pdf["log_radius"] = np.log10(pdf["planet_radius"])
pdf["log_temp"] = np.log10(pdf["star_temp"])

# Model: log(P) = a * log(R) + b * log(T) + c
X = pdf[["log_radius", "log_temp"]]
y = pdf["log_period"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
print(f"R²: {r2:.4f}")
print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
print(f"Intercept: {model.intercept_:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Result:**  
# MAGIC The R² value is around 0.19, showing a weak but noticeable relationship.  
# MAGIC Larger planets tend to have longer orbits, and hotter stars often have planets that move faster around them.  
# MAGIC This confirms that while our model is simple, it still captures traces of the physical laws that govern planetary motion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Visualizing Model Accuracy
# MAGIC
# MAGIC The scatter plot compares the observed and predicted orbital periods.  
# MAGIC Points close to the red line show where the model aligns well with reality.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.scatter(y_test, pred, alpha=0.6)
plt.xlabel("Observed log(Orbital Period)")
plt.ylabel("Predicted log(Orbital Period)")
plt.title("Kepler’s Law Validation with Databricks + NASA Exoplanet Data")
plt.show() 
plt.savefig("/Workspace/Shared/kepler_validation.png", dpi=150)

# COMMAND ----------

# MAGIC %md
# MAGIC Even with limited information, we can see that hotter stars tend to have faster-orbiting planets, while cooler stars host slower ones.  
# MAGIC This reflects the same pattern that Kepler described centuries ago.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Starting from raw telescope data, we prepared, visualized, and modeled thousands of exoplanets in just a few lines of Python.  
# MAGIC Databricks made it easy to explore scientific data interactively and test physical patterns quickly.
# MAGIC
# MAGIC **Key takeaway:**  
# MAGIC Even across hundreds of alien worlds, the same cosmic principles apply.  
# MAGIC Data reveals the order behind the stars.
# MAGIC This approach could easily be extended to include mass and distance for deeper habitability modeling.
# MAGIC