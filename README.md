# Flood Monitoring & Prediction with an LSTM

Predicts river water level 1, 2 and 3 steps (hours) ahead from the previous 24 hours of
rainfall, water level, soil moisture and temperature.

```
flood_lstm/
├── config.py          # all settings (paths, columns, lookback, horizons, hyper-parameters)
├── generate_data.py   # synthetic dataset generator -> data/flood_data.csv
├── preprocess.py      # clean, chronological split, MinMaxScaler, sliding windows
├── model.py           # Keras LSTM architecture
├── train.py           # train, plot loss, save .keras, evaluate (RMSE / MAE / R2)
├── test_predict.py    # load model, predict on new data, PASS/FAIL checks
└── requirements.txt
```

## Setup in VS Code (Windows / macOS / Linux)

1. Install Python 3.10-3.12 and the VS Code **Python** extension.
2. `File > Open Folder...` and choose `flood_lstm`.
3. Open a terminal: **Terminal > New Terminal**.
4. Create the virtual environment:
   - Windows: `py -3.12 -m venv .venv` (or `python -m venv .venv`)
   - macOS/Linux: `python3 -m venv .venv`
5. Activate it:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
     (if blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry)
   - Windows cmd: `.venv\Scripts\activate.bat`
   - macOS/Linux: `source .venv/bin/activate`
6. Select the interpreter: `Ctrl+Shift+P` > **Python: Select Interpreter** > choose `.venv`.
7. `python -m pip install --upgrade pip`, then `pip install -r requirements.txt`.

## Run order

```bash
python generate_data.py      # 1. create data/flood_data.csv (skip when using your real CSV)
python preprocess.py         # 2. optional smoke test: prints array shapes
python train.py              # 3. train + evaluate; writes models/ and outputs/
python test_predict.py       # 4. PASS/FAIL checks on unseen data
```

Quick trial: `python train.py --epochs 5`. Show plot windows: `python train.py --show`.

## Using your real thesis dataset

1. Put the CSV in `data/` and set `RAW_CSV_PATH` in `config.py`.
2. Map your column names in `RENAME_MAP` (`config.py`).
3. Set `DATA_FREQ` (e.g. `"10min"`) and choose `HORIZONS` in *steps*.
4. Delete `models/` (old scalers) and re-run `train.py`.
5. In `test_predict.py`, replace the synthetic generator inside `load_new_data()` with your newer data.
