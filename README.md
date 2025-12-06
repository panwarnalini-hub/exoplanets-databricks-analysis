# Exploring NASA Exoplanets on Databricks

This repository analyzes thousands of confirmed exoplanets using Databricks Community Edition. The project uses data from NASA's Exoplanet Archive to study patterns in planetary physics and investigate whether the same rules observed in our Solar System generalize beyond it.

## Source Notebook

- [NASA Exoplanet Archive](NASA%20Exoplanet%20Archive.py)
- [HTML Export (visual view)](NASA%20Exoplanet%20Archive.html)

---

## Overview

The notebook demonstrates an end-to-end scientific workflow on a single platform:

- Loading and inspecting NASA exoplanet data
- Cleaning and standardizing physical attributes
- Engineering a basic indicator of Earth-like conditions
- Exploring discovery trends using SQL
- Visualizing orbital characteristics for pattern detection
- Testing a simplified model relating size, temperature, and orbital period

Spark is used for scalable transformation, SQL for analysis, and scikit-learn for a small physical validation exercise.

---

## Dataset Description

The dataset originates from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and contains confirmed planets discovered primarily by Kepler, TESS, and other space telescopes. Each record includes:

| Field | Description |
|-------|-------------|
| Planet name | Identifier for the exoplanet |
| Radius | Earth radii |
| Orbital period | Days |
| Host star temperature | Kelvin |
| Discovery year | Year of confirmation |
| Detection method | Technique used for discovery |

---

## Habitability Scoring (Heuristic)

A simple feature is included to highlight planets with potentially familiar characteristics. This "habitability score" increases when:

- Planet radius is near 1 Earth radius
- Host star temperature lies in the moderate range (approximately 4800–6200 K)

> **Note:** This score is not a scientific statement about habitability. It is a lightweight heuristic for exploratory filtering.

---

## Kepler-Style Relationship Test

A small regression experiment evaluates whether orbital period can be approximated using only:

- Planetary radius
- Host star temperature

A logarithmic transform captures the multiplicative nature of orbital mechanics. The resulting model achieves an **R² of approximately 0.19**, indicating a weak but non-random physical relationship. Even coarse measurements reveal hints of the dynamics described by Kepler's laws.

---

## Visual Exploration

The notebook applies dimensional filtering to focus on planets with orbital periods under 1000 days, which roughly corresponds to zones where liquid water may be feasible. Scatter plots reveal clusters of rocky planets in these time ranges, suggesting where further research could be prioritized.

---

## How to Reproduce

### Running on Databricks (recommended)

1. Create a free [Databricks Community Edition](https://community.cloud.databricks.com/) account
2. Import `NASA Exoplanet Archive.py` into your workspace
3. Upload the dataset into the default tables volume
4. Attach an active cluster and execute cells sequentially

### Running Locally

**Requires:** PySpark, pandas, numpy, scikit-learn

Update the dataset path in the `spark.read.csv()` call before running.

---

## Repository Structure

```
├── NASA Exoplanet Archive.py    # Databricks source notebook
├── NASA Exoplanet Archive.html  # HTML export for visual reference
├── images/
│   └── databricks_notebook_view.png           # Databricks interface capture
└── README.md
```

---

## Planned Extensions

- [ ] Incorporate planet mass and orbital distance to better approximate stellar flux
- [ ] Add time-series analysis of discoveries by mission and instrument
- [ ] Build a reproducible SQL or dashboard layer
- [ ] Evaluate habitability using established models rather than heuristics
- [ ] Benchmark different predictive models with physical features

---

## Submission Context

This work began as a submission in a Databricks Free Edition learning challenge. The objective was to demonstrate data exploration, scientific reasoning, and basic modeling using freely available space data.

It remains a self-contained example of using modern data tools for research-style investigation.

---

## Core Idea

> Despite enormous variation across planetary systems, there are still recognizable structures in the data. The same physical principles that shaped Earth also appear to govern distant worlds. Data analysis allows us to detect that order.

---

## License

*Add your preferred license here*
