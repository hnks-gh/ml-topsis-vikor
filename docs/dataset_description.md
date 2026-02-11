# Dataset Description

## Overview
This document provides a comprehensive description of the annual CSV data files, which contain panel data of 28 sub-criteria scores for the governance index of 63 provinces in Vietnam from 2011-2024 (14 years). Each year's data is stored in a separate CSV file.

---

## File Information

- **File Path**: `data/YYYY.csv` (where YYYY = 2011 to 2024)
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Files**: 14 separate files (2011.csv, 2012.csv, ..., 2024.csv)
- **Rows per File**: 64 (including header)
- **Data Rows per File**: 63
- **Columns**: 30 (Year, Province, and 28 sub-criteria)

---

## Dataset Structure

### Temporal Coverage
- **Start Year**: 2011
- **End Year**: 2024
- **Duration**: 14 years
- **Frequency**: Annual observations

### Spatial Coverage
- **Geographic Region**: Vietnam
- **Administrative Units**: 63 provinces
- **Province Identifiers**: P01 through P63

### Total Observations
- **Formula**: 63 provinces × 14 years = 882 observations
- **Distribution**: 63 provinces per year file × 14 year files
- **Data Structure**: Separate CSV file for each year

---

## Column Specifications

### 1. Temporal Identifier
- **Column Name**: `Year`
- **Data Type**: Integer
- **Range**: 2011-2024
- **Purpose**: Identifies the year of observation

### 2. Spatial Identifier
- **Column Name**: `Province`
- **Data Type**: String (Categorical)
- **Format**: P## (where ## is a two-digit number)
- **Range**: P01 to P63
- **Purpose**: Unique identifier for each province

### 3. Sub-Criteria (28 columns)
- **Column Names**: SC01, SC02, SC03, ..., SC28
- **Data Type**: Float (Decimal numbers)
- **Value Range**: Approximately 0.00 to 1.00
- **Total Sub-Criteria**: 28 governance indicators
- **Purpose**: Provincial governance and public administration index sub-criteria

#### Hierarchical Structure
The 28 sub-criteria are organized under 8 main criteria (not present in dataset):

| Main Criteria | Sub-Criteria Range | Count | Description |
|---------------|-------------------|-------|-------------|
| Criteria 1: Participation | SC01 - SC04 | 4 | Civic engagement and electoral participation |
| Criteria 2: Transparency of Local Decision-making | SC05 - SC08 | 4 | Information access and budget transparency |
| Criteria 3: Vertical Accountability | SC09 - SC11 | 3 | Government responsiveness and justice access |
| Criteria 4: Control of Corruption | SC12 - SC15 | 4 | Anti-corruption measures and equity |
| Criteria 5: Public Administrative Procedures | SC16 - SC18 | 3 | Administrative efficiency and procedures |
| Criteria 6: Public Service Delivery | SC19 - SC22 | 4 | Healthcare, education, and infrastructure |
| Criteria 7: Environmental Governance | SC23 - SC25 | 3 | Environmental protection and quality |
| Criteria 8: E-Governance | SC26 - SC28 | 3 | Digital government and internet access |

**Note**: Each sub-criterion (SC01-SC28) represents a specific aspect of provincial governance. The values are normalized to a 0-1 scale, where:
- Higher values (closer to 1.00) indicate better performance on that sub-criterion
- Lower values (closer to 0.00) indicate poorer performance on that sub-criterion

---

## Data Schema

```
YYYY.csv (e.g., 2011.csv, 2012.csv, ..., 2024.csv)
├── Year          : int   [2011-2024]
├── Province      : str   [P01-P63]
├── SC01          : float [0.00-1.00]
├── SC02          : float [0.00-1.00]
├── SC03          : float [0.0-1.00]
├── ...           : ...   [...]
└── SC28          : float [0.00-1.00]
```

---

## File List

The dataset consists of 14 separate CSV files located in the `data/` directory:

- `2011.csv` - Data for year 2011
- `2012.csv` - Data for year 2012
- `2013.csv` - Data for year 2013
- `2014.csv` - Data for year 2014
- `2015.csv` - Data for year 2015
- `2016.csv` - Data for year 2016
- `2017.csv` - Data for year 2017
- `2018.csv` - Data for year 2018
- `2019.csv` - Data for year 2019
- `2020.csv` - Data for year 2020
- `2021.csv` - Data for year 2021
- `2022.csv` - Data for year 2022
- `2023.csv` - Data for year 2023
- `2024.csv` - Data for year 2024

Each file contains 63 rows (one per province) plus a header row.

---

## Additional Resources

For detailed variable descriptions, provincial names, and sub-criteria definitions, please refer to the **codebook files** located in the `data/codebook` directory:

- **`codebook_provinces.csv`** - Province codes (P01-P63) and their full names
- **`codebook_criteria.csv`** - Main criteria codes (C01-C08) and descriptions
- **`codebook_subcriteria.csv`** - Sub-criteria codes (SC01-SC28), names, and their parent criteria mapping