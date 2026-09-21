# Predicting car prices with linear regression

A linear regression model built from scratch with NumPy — no scikit-learn — that predicts a car's price
from its specifications. The model is solved directly from the normal equation, `w = (XᵀX)⁻¹Xᵀy`, which
means every numerical thing that can go wrong is visible instead of hidden behind a library call.

That turned out to be the interesting part. **This notebook is mostly a story about the model breaking**,
three times, in three different disguises, for the same underlying reason.

**Final result:** RMSE **0.4528** on the log price, measured on a test set that was touched exactly once,
at the end. In plain terms, the typical prediction lands about **25% away** from the real price.

---

## The one rule

Every failure in this notebook is the same rule being broken:

> **No column of the input matrix may be an exact recipe of the other columns.**

If one is, `XᵀX` has no inverse, the weights are not unique, and the model is meaningless. What makes it
worth a whole notebook is that it rarely announces itself:

| section | the redundant column | how it broke |
|---|---|---|
| 6 | `has_2_doors + has_3_doors + has_4_doors` equals the bias column | silent garbage — RMSE 16.5 |
| 9 | `popularity` is a per-make constant, so the make columns can rebuild it exactly | silent garbage — RMSE 95.2 |
| 10 | `make_is_genesis` is all zeros, because that split had no Genesis to train on | `LinAlgError` |

Only the third one raised an error. The first two returned confident, entirely wrong numbers — which is the
uncomfortable lesson: a broken linear regression looks exactly like a working one until you check the
condition number.

Section 11 fixes the third case with ridge regression, and the honest finding there is that ridge does
**not** make this model more accurate (0.4543 either way on the splits where both work). What it buys is
that the model trains on all 40 random splits instead of 31.

## Results

Each model scored on the validation set (seed 0):

| # | model | data | validation RMSE | what it shows |
|---|---|---|---|---|
| 1 | base features | `df` | 0.7734 | starting point |
| 2 | + `car_age` | `df` | 0.5301 | the age matters a lot |
| 3 | + 3 door columns | `df` | 0.5279 | barely changes; only works thanks to 6 cars |
| 4 | + 3 door columns | `df_corrected` | 16.5354 | dummy variable trap: broken |
| 5 | + `car_age` | `df_corrected` | 0.5146 | new baseline after removing the 6 cars |
| 6 | + 2 door columns | `df_corrected` | 0.5143 | fixed, but the doors add ~nothing |
| 7 | + 48 make columns | `df_corrected` | 111.2961 | dummy variable trap again |
| 8 | + 47 make columns | `df_corrected` | 95.2455 | still broken: `popularity` is a recipe |
| 9 | + 47 make columns, no `popularity` | `df_corrected` | **0.4478** | the make is a real improvement |
| 10 | … and ridge, r = 0.1 | `df_corrected` | 0.4478 | same accuracy, but always trainable |

**Final model on the held-out test set: 0.4528.**

Two features were tested and reported honestly rather than quietly dropped:

- **Number of doors** — measured across 40 random splits, the average improvement is −0.00001. It does not
  help. Section 8 says so.
- **Make** — +0.068 across the same 40 splits, helping on 30 of the 31 that could be trained. It does help.

## What's in the repo

```
car-price-prediction.ipynb   the whole project, 13 sections, runs top to bottom
data.csv                     the dataset (also auto-downloaded if missing)
README.md                    this file
```

The notebook is written to be read in order. Every number quoted in its markdown was checked against a
fresh Restart & Run All.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install numpy pandas jupyter
jupyter notebook car-price-prediction.ipynb
```

Written against Python 3.12, NumPy 2.5 and pandas 3.0. The pandas 3 part matters: copy-on-write is always
on, so the `df["col"].fillna(0, inplace=True)` idiom that most car-price tutorials use silently does
nothing here.

## The data

The Kaggle *Car Features and MSRP* dataset — 11,914 cars, 16 columns, model years 1990–2017, with the
manufacturer's suggested retail price as the target. `data.csv` is included for reproducibility; the
notebook's first cell also fetches it from
[`mlbookcamp-code`](https://github.com/alexeygrigorev/mlbookcamp-code) if it is missing.

The target is log-transformed (`np.log1p`), because prices run from $2,000 to $2,065,902 and a handful of
supercars would otherwise dominate every error.

## Known limitations

Listed in full at the end of the notebook. The main ones:

- A single ridge penalty `r` is applied to columns on wildly different scales, so it leans much harder on
  the 0/1 make columns than on `engine_hp`. The columns should be rescaled first.
- `fillna(0)` was applied to the whole table early on. Section 6 catches the damage to `number_of_doors`,
  but the same fill also filed 20 Mazda RX-7s and RX-8s under "0 cylinders" — which in this dataset means
  *electric*. They are rotary engines.
- 1,036 cars have an MSRP of exactly $2,000, every one of them built between 1990 and 2000. That is a
  placeholder, not a price, and it distorts anything the model learns about older cars.
- Model selection rests on one validation split. Cross-validation would be the better tool.

## Credits

Follows the car-price project from chapter 2 of Alexey Grigorev's *Machine Learning Bookcamp* / ML Zoomcamp,
with the analysis extended well past the original — the `popularity` collinearity, the all-zero-column
failure, the multi-seed feature tests and the ridge section are not in the source material.
