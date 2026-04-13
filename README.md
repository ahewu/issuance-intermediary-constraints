# Issuance and Intermediary Constraints

This repository implements an empirical pipeline for testing whether volatility-driven declines in corporate bond issuance are better explained by intermediary constraints or by real-options timing behavior.

## Project structure
- `src/extract`: raw data pulls
- `src/clean`: source-specific cleaning
- `src/link`: identifier mapping
- `src/construct`: variable construction
- `src/analysis`: regressions and tables
- `scripts/`: ordered pipeline entry points

## Setup
1. Create virtual environment
2. Install requirements
3. Add WRDS username to `.env` or local config
4. Run:
   `python scripts/00_project_smoke_test.py`

## Main data sources
- Mergent FISD
- Compustat
- TRACE
- Treasury/MOVE data

## Main empirical outputs
- baseline issuance regressions
- liquidity AR(1)
- mechanism/mediation regressions
- fragility vs flexibility interaction tests