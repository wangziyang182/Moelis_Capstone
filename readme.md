# Moelis Capstone Project

## Members

- Junhao Wang 
- Dazun Sun 
- Jie Zheng
- Zhiyuan Zhao
- Michael She

## Usage

Before running the model:

- Copy `Data/FactSet_Campaign v8.xlsx` from the Google Drive to `data/factset_campaign_v9.xlsx` in your local repository
- Copy `Data/FactSet_Pricing.txt` from the Google Drive to `data/factset_pricing.txt` in your local repository

To run the data pipeline:

- Run `python main.py`, which does nothing right now but will at some point.

To run individual model notebooks, open Jupyter Lab and open any of the notebooks in the `/notebook` folder. For example the primary model notebooks are:

- `campaign_primary_objective.ipynb`
- `campaign_proxy_result.ipynb`
- `campaign_return.ipynb`

Make sure to open Jupyter Lab or Jupyter Notebook in the root ./ folder, not within the notebook/ folder. This ensures that all file paths referenced are with respect to the root folder.