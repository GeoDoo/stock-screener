# Magic Formula Stock Screener

This project implements Joel Greenblatt's Magic Formula investing strategy using SimFin data. The screener ranks stocks based on a combination of:
- Return on Capital (ROC)
- Earnings Yield (EY)

## Implementation Notes

This is a simplified implementation of the Magic Formula strategy. For the most accurate results, consider using the official Magic Formula website which uses Compustat Point-in-Time (PIT) data.

### Key Differences from Official Implementation

1. Data Source:
   - This implementation: Uses SimFin data (current/restated values)
   - Official website: Uses Compustat PIT data (historical point-in-time values)

2. Data Quality:
   - This implementation: Uses current financial statements
   - Official website: Uses historical data as it was reported
   - Impact: May include look-ahead bias in backtesting

3. Data Freshness:
   - This implementation: Uses latest available quarterly/annual data
   - Official website: Uses PIT data with proper historical context

## Requirements

- Python 3.7+
- SimFin data files in `simfin_data` directory:
  - `us-companies.csv`
  - `us-income-annual.csv`
  - `us-balance-annual.csv`
  - `us-shareprices-daily.csv`
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

3. Place SimFin data files in the `simfin_data` directory

## Usage

Run the screener:
```bash
python magic_formula.py
```

The script will:
1. Load financial data from local SimFin CSV files
2. Calculate ROC and Earnings Yield for each stock
3. Apply filters:
   - Market cap > $50M
   - Positive EBIT
   - Positive Invested Capital
   - Positive Enterprise Value
   - Data within last 2 years
4. Rank stocks based on combined metrics
5. Output top 50 stocks to console and CSV file

## Data Sources

This implementation uses SimFin data files for:
- Company information
- Annual financial statements
- Daily share prices

For more accurate results, consider:
1. Using Compustat PIT data (requires academic/professional access)
2. Implementing SEC EDGAR data parsing
3. Using multiple data sources for validation

## Limitations

1. No Point-in-Time Data:
   - Uses current/restated values
   - May include look-ahead bias
   - Less accurate for historical analysis

2. Data Freshness:
   - Depends on SimFin data updates
   - May lag behind market data
   - Uses annual data instead of quarterly

3. Filtering:
   - Basic implementation of Greenblatt's criteria
   - May miss some edge cases
   - Limited to available data fields

## Future Improvements

1. Data Sources:
   - Implement SEC EDGAR data parsing
   - Add multiple data source validation
   - Track data revisions

2. Calculations:
   - Add more sophisticated filtering
   - Implement industry-specific adjustments
   - Add risk metrics

3. Output:
   - Add more detailed analysis
   - Include historical performance
   - Add visualization tools

## ⚠️ WARNING

**This app is not using Compustat PIT data. Use at your own risk!**

The results may differ significantly from the official Magic Formula website due to:
- Lack of point-in-time data
- Potential look-ahead bias
- Different data sources and calculations
- Missing historical revisions and restatements
