# eThekwini Metro - 04 Nov 2026 Local Government Election Forecast

Voting-district-level forecast built on the 2016 and 2021 results, following the ML lifecycle
(problem definition, data audit, features, training, validation, prediction, deployment).

## Outputs
1. Largest party, seat projection and minimal winning coalitions
2. Projected leader in three selected wards
3. Projected vote share per relevant party (with 80% ranges)
4. Projected voter turnout (number of voters)

## Repository layout
| File | Purpose |
|---|---|
| `ElectionForecast.ipynb` | Self-contained notebook: pipeline code + results + charts |
| `election_forecast.py` | Same pipeline as a module (used by the dashboard) |
| `app.py` | Streamlit dashboard |
| `data/` | `2016_Results.csv` (2016), `ETH.csv` (2021), `Pre_Election_Voting_Districts_2026.csv` |

## Run
```bash
pip install -r requirements.txt
jupyter notebook ElectionForecast.ipynb   # Restart & Run All
streamlit run app.py                                # dashboard
```
Trained models are cached with joblib in `models/` (git-ignored).

## Method
Votes are converted to centred log-ratios per voting district. Forecast =
2021 position + persistence x metro-wide 2016-21 swing + a gradient-boosting model of local deviations.
Uncertainty comes from Monte-Carlo simulation. Validation: ward-grouped 5-fold backtest of 2016 -> 2021
against "no change" and "uniform swing" baselines.

## Dataset usability
- 2016 and 2021 results: used for training.
- 2026 districts file: contains no votes; used as the prediction frame (districts, wards, urban/rural).

## Limitations
- Only two past elections: whether the 2016-21 swing continues is a scenario assumption (`persistence`), not learned.
- No 2024 results, so parties that emerged after 2021 (e.g. MK) are not modelled; use the `emerging_share` scenario.
- 2026 registered-voter numbers are not in the data (`reg_growth` defaults to 0%).
- Council size assumed to be 2 x wards; independents are pooled with "other".
- Coalitions are seat arithmetic only.
