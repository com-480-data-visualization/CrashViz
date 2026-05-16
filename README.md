# Project of Data Visualization (COM-480)

| Student's name | SCIPER |
| -------------- | ------ |
| Valentin Dupraz | 315995 |
| Yahya Kerem Molla | 389791 |
| Jason Santangelo | 312202 |

[Milestone 1](#milestone-1) • [Milestone 2](#milestone-2) • [Milestone 3](#milestone-3)

## Milestone 1 (20th March, 5pm)

**10% of the final grade**

You can find our report for Milestone 1 [here](Milestone1/Milestone_1.pdf).

The associated notebook is [here](Milestone1/dataset_download_eda.ipynb).

## Milestone 2 (17th April, 5pm)

**10% of the final grade**

You can find our report for Milestone 2 [here](Milestone2/Milestone_2.pdf).

The associated htlm is [here](Milestone2/index.html).

To view the live website skeleton directly in your browser, click [here](https://com-480-data-visualization.github.io/CrashViz/Milestone2/index.html).


## Milestone 3 (29th May, 5pm)

**80% of the final grade**

You can find our process book [here](Milestone3/ProcessBook.pdf) and our screencast [here](Milestone3/screencast.mp4).
 
The entry point of the website is [here](Milestone3/welcome.html).
 
To view the live website directly in your browser, click [here](https://com-480-data-visualization.github.io/CrashViz/Milestone3/welcome.html).
 
### About CrashViz
 
CrashViz is a scroll-driven exploration of 25 years of cross-asset correlation regimes (2000 to 2025). Three layers, navigated in sequence:
 
```
welcome.html  →  index.html  →  globe.html
                    ▲              │
                    └──[BACK/ESC]──┘
```
 
- **`welcome.html`** : cinematic intro with regime-reactive ambient audio
- **`index.html`** : force-directed correlation graph, scroll the timeline through 2000 to 2025
- **`globe.html`** : per-asset geographic dossier (top producers and consumers)
Timeline position persists in `sessionStorage` across the index ↔ globe round trip.
 
### Quick start
 
```bash
cd Milestone3
python compute_correlations.py   # optional, correlation_data.json is committed
python -m http.server 8000
```
 
Then open `http://localhost:8000/welcome.html`.
 
### Files
 
```
Milestone3/
├── welcome.html  ·  index.html  ·  globe.html       # 3-layer app
├── compute_correlations.py                          # 63-day rolling Pearson → JSON
├── market_data_2000_2025.csv                        # cleaned daily closes
└── correlation_data.json                            # generated, ~250 monthly matrices
```
 
### Data sources
 
- Prices: Yahoo Finance via `yfinance`
- Producer / consumer shares: IEA 2023, USGS 2024, World Gold Council 2023, ICSG 2023, Silver Institute 2024, USDA Oct 2024
- Basemap: `world-atlas@2/countries-110m.json`
