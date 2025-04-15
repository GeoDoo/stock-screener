# Magic Formula Stock Screener

This project implements Joel Greenblatt's Magic Formula investing strategy using SimFin data. The screener ranks stocks based on a combination of:
- Return on Capital (ROC)
- Earnings Yield (EY)

## Requirements

- Python 3.7+
- SimFin account (free tier available)
- Required Python packages (see requirements.txt)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Sign up for a free SimFin account at https://simfin.com/

4. Set your SimFin API key as an environment variable:
```bash
export SIMFIN_API_KEY='your-api-key-here'
```

## Usage

Run the screener:
```bash
python magic_formula_simfin.py
```

The script will:
1. Download and cache financial data from SimFin
2. Calculate ROC and Earnings Yield for each stock
3. Rank stocks based on combined metrics
4. Output top 30 stocks to console and CSV file

## Implementation Details

- Minimum market cap filter: $50M
- Uses quarterly financial data for recent results
- Excludes stocks with negative/zero EBIT
- Automatically caches financial data locally
- Outputs results to a dated CSV file

## Data Sources

This implementation uses SimFin (https://simfin.com/) for financial data. SimFin provides:
- Standardized financial statements
- Quarterly and annual data
- Market metrics (price, market cap, etc.)
- Automatic data caching
