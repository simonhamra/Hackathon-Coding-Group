# 🧠 Team Coding Rookie - Intertemporal Choice Analysis

> **Data Hackathon 2026** - Analyzing when and why people choose smaller-sooner vs. larger-later rewards across 800k+ trials from multi-study behavioral data.

🔗 **[View our findings →](https://apoorvverma.github.io/coding-rookie/)**

---

## 📁 Project Structure

```
coding-rookie/
├── Data_Hackathon_Analysis.ipynb   # Full notebook (run end-to-end in Colab/Jupyter)
├── data_hackathon_analysis.py      # Equivalent Python script
├── images/                         # Exported charts used in the GitHub Pages site
├── index.html                      # GitHub Pages presentation site
├── requirements.txt                # Python dependencies
└── README.md
```

## 🚀 Quick Start

Our cleaned data set we used for all modelings and data exploration ( can also execute data cleaning part of "Data_Hackathon_Analysis.ipynb" to get it if original data set downloaded) : https://drive.google.com/file/d/11Rh1NIjVs4byME5EBziOIxH-95TAeKYL/view?usp=sharing

Original data set : https://drive.google.com/file/d/1KN6amaHM70ke327ou-CIjI_XWdAgMkCU/view?usp=sharing

```bash
# 1. Clone the repo
git clone https://github.com/apoorvverma/coding-rookie.git
cd coding-rookie

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the raw dataset in the project root
#    Expected file: all_data.csv

# 4. Run the full pipeline
python data_hackathon_analysis.py
```

Alternatively, open `Data_Hackathon_Analysis.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter and run all cells sequentially.

## 🔬 Analytical Pipeline

### 1. Data Preparation

- Loaded raw multi-study CSV (~800k+ trials, 20 columns)
- Enforced binary `choice` (0 = SS, 1 = LL), coerced numeric types
- Dropped rows missing core task columns (`ss_value`, `ss_time`, `ll_value`, `ll_time`)
- Removed subject-level and trial-level exclusions flagged by original authors
- Winsorized response times at the 99.5th percentile (capped, not deleted)

### 2. Feature Engineering

- `value_diff` = LL value − SS value
- `time_diff_days` = LL delay − SS delay
- `reward_ratio` = LL value / SS value (positive denominator only)
- `age_group` — binned into 12 granular brackets to avoid masking distributional patterns

### 3. Exploratory Analysis


| Question                   | Method                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------- |
| Delay structure (SS vs LL) | Count plots + boxplots by time bucket                                                 |
| Overall choice balance     | Pie chart of SS/LL split                                                              |
| Reward ratio tipping point | Decile-binned LL rate curve → **~1.29× threshold**                                    |
| Country-level variation    | Horizontal bar charts (filtered to n > 500)                                           |
| Age effects on choice & RT | Bar plots + boxplots by granular age group                                            |
| RT vs reward ratio         | Scatter + regression (Pearson r ≈ 0)                                                  |
| Context variables          | LL rate by procedure, incentivization, online/lab, time pressure, presentation format |


### 4. Feature Ranking

- Mutual information (MI) with binary choice across 12 numeric + 4 categorical features
- Subsampled to 200k trials for computational efficiency (seed = 42)

## 📊 Key Findings

- **Tipping ratio ~1.29×** — LL adoption crosses 50% when the larger option is ≥ 29% more valuable
- **Task design dominates** — procedure type explains ~30pp spread in LL rates, more than any demographic
- **Online ≠ lab** — digital settings trend ~8pp more impatient; benchmark accordingly
- **Age slows RT, not patience linearly** — older groups are slower and more variable, but not uniformly less patient
- **MI ranking** — reward amounts/delays rank highest (task structure); context variables matter for causal interpretation

## 🛠️ Tech Stack

- **Python 3.10+**
- pandas, NumPy — data wrangling
- matplotlib, seaborn — visualization
- scikit-learn — mutual information feature ranking

## 👥 Team
- Tran Phuong Ngoc Ngo
- Sudhir, Antara
- Hamra Marchand, Simon
- Verma, Apoorv

**Team Coding Rookie**

---

Results presented at [apoorvverma.github.io/coding-rookie](https://apoorvverma.github.io/coding-rookie/)