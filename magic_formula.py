import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Setup logging when module is imported
setup_logging()

def download_simfin_data():
    """Make ONE single API call to SimFin to get ALL data."""
    data_dir = Path('simfin_data')
    data_dir.mkdir(exist_ok=True)
    
    # Check if we already have the data
    if (data_dir / 'us-companies.csv').exists():
        logging.info("SimFin data already exists locally")
        return
    
    # Get API key from environment variable
    api_key = os.getenv('SIMFIN_API_KEY')
    if not api_key:
        raise ValueError("SIMFIN_API_KEY not found in .env file")
    
    # ONE single API call to get ALL data
    logging.info("Making ONE API call to SimFin to get ALL data...")
    url = "https://api.simfin.com/v1/data/all"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Save all data to local files
    data = response.json()
    
    # Save companies data
    pd.DataFrame(data['companies']).to_csv(data_dir / 'us-companies.csv', index=False)
    logging.info("Saved companies data")
    
    # Save income statements
    pd.DataFrame(data['income']).to_csv(data_dir / 'us-income.csv', index=False)
    logging.info("Saved income statements")
    
    # Save balance sheets
    pd.DataFrame(data['balance']).to_csv(data_dir / 'us-balance.csv', index=False)
    logging.info("Saved balance sheets")
    
    # Save price data
    pd.DataFrame(data['prices']).to_csv(data_dir / 'us-shareprices-daily.csv', index=False)
    logging.info("Saved price data")

def fetch_all_stock_data():
    """
    Load stock data from local SimFin CSV files.
    Returns tuple of DataFrames: (companies_df, income_df, balance_df, prices_df)
    """
    data_dir = Path('simfin_data')
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please run download_simfin_data() first.")
    
    try:
        # Load companies data
        companies_path = data_dir / 'us-companies.csv'
        if not companies_path.exists():
            raise FileNotFoundError(f"Companies file not found at {companies_path}")
        companies_df = pd.read_csv(companies_path, sep=';', encoding='utf-8')
        logging.info(f"Loaded {len(companies_df)} companies")
        
        # Load income statement data
        income_path = data_dir / 'us-income-annual.csv'
        if not income_path.exists():
            raise FileNotFoundError(f"Income statements file not found at {income_path}")
        income_df = pd.read_csv(income_path, sep=';', encoding='utf-8')
        logging.info(f"Loaded {len(income_df)} income statements")
        
        # Load balance sheet data
        balance_path = data_dir / 'us-balance-annual.csv'
        if not balance_path.exists():
            raise FileNotFoundError(f"Balance sheets file not found at {balance_path}")
        balance_df = pd.read_csv(balance_path, sep=';', encoding='utf-8')
        logging.info(f"Loaded {len(balance_df)} balance sheets")
        
        # Load prices data
        prices_path = data_dir / 'us-shareprices-daily.csv'
        if not prices_path.exists():
            raise FileNotFoundError(f"Share prices file not found at {prices_path}")
        prices_df = pd.read_csv(prices_path, sep=';', encoding='utf-8')
        logging.info(f"Loaded {len(prices_df)} price records")
        
        # Convert date columns to datetime
        date_columns = ['Report Date', 'Publish Date', 'Restated Date']  # Removed 'Fiscal Period' and 'Fiscal Year'
        for df in [income_df, balance_df]:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Handle Fiscal Year separately (it's just a year)
            if 'Fiscal Year' in df.columns:
                df['Fiscal Year'] = df['Fiscal Year'].astype(int)
        
        if 'Date' in prices_df.columns:
            prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        
        return companies_df, income_df, balance_df, prices_df
        
    except pd.errors.EmptyDataError:
        logging.error("One or more CSV files are empty")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def get_sec_tickers():
    """Get ticker information from SEC and save locally."""
    data_dir = Path('simfin_data')
    sec_file = data_dir / 'sec_tickers.csv'
    
    # If file exists and is less than 24 hours old, use it
    if sec_file.exists():
        file_age = datetime.now() - datetime.fromtimestamp(sec_file.stat().st_mtime)
        if file_age.days < 1:
            logging.info("Using cached SEC tickers file")
            return pd.read_csv(sec_file)
    
    # Download fresh data from SEC
    logging.info("Downloading fresh SEC tickers data...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Convert to DataFrame
    tickers_df = pd.DataFrame.from_dict(response.json()['data'], orient='records')
    
    # Rename columns
    tickers_df = tickers_df.rename(columns={
        'cik': 'CIK',
        'ticker': 'Ticker',
        'name': 'Company Name SEC',
        'exchange': 'Exchange'
    })
    
    # Convert CIK to string with leading zeros
    tickers_df['CIK'] = tickers_df['CIK'].astype(str).str.zfill(10)
    
    # Save to local file
    data_dir.mkdir(exist_ok=True)
    tickers_df.to_csv(sec_file, index=False)
    logging.info(f"Saved SEC tickers to {sec_file}")
    
    return tickers_df

def get_common_tickers():
    """Get a mapping of common company names to their tickers."""
    return {
        'Crocs, Inc.': 'CROX',
        'AMC Networks Inc': 'AMCX',
        'CHENIERE ENERGY INC': 'LNG',
        'CONSOL Energy Inc.': 'CEIX',
        'CRAWFORD & CO': 'CRD.A',
        'China Automotive Systems, Inc.': 'CAAS',
        'Consensus Cloud Solutions, Inc.': 'CCSI',
        'Cross Country Healthcare, Inc.': 'CCRN',
        'EVOLUTION PETROLEUM CORP': 'EPM',
        'FEDERATED INVESTORS INC /PA/': 'FHI',
        'Fox Corp': 'FOX',
        'Freedom Holding Corp.': 'FRHC',
        'GULFPORT ENERGY CORP': 'GPOR',
        'Gambling.com Group Limited': 'GAMB',
        'Garrett Motion Inc.': 'GTX',
        'H&R BLOCK INC': 'HRB',
        'HUDSON TECHNOLOGIES INC /NY': 'HDSN',
        'Harmony Biosciences Holdings, Inc.': 'HRMY',
        'Heritage Global Inc.': 'HGBL',
        'InterDigital, Inc.': 'IDCC',
        'J.Jill, Inc.': 'JILL',
        'JAKKS PACIFIC INC': 'JAKK',
        'Legacy Housing Corp': 'LEGH',
        'MCBC Holdings, Inc.': 'MCFT',
        'MCCORMICK & CO INC': 'MKC',
        'MEDIFAST INC': 'MED',
        "Nathan's Famous, Inc.": 'NATH',
        'Natural Resource Partners L.P.': 'NRP',
        'NetApp, Inc.': 'NTAP',
        'OMNICOM GROUP INC.': 'OMC',
        'Premier Inc': 'PINC',
        'Priority Technology Holdings, Inc.': 'PRTH',
        'RCM Technologies, Inc.': 'RCMT',
        'REX American Resources Corporation': 'REX',
        'Semler Scientific, Inc.': 'SMLR',
        'Smith A O Corp': 'AOS',
        'Star Group, L.P.': 'SGU',
        'Stellantis N.V.': 'STLA',
        'StoneCo Ltd.': 'STNE',
        'TRAVELZOO': 'TZOO',
        'UNIVERSAL LOGISTICS HOLDINGS, INC.': 'ULH',
        'Virtu Financial, Inc.': 'VIRT',
        'Voyager Therapeutics, Inc.': 'VYGR',
        'WABASH NATIONAL CORP /DE': 'WNC',
        'Western Union Co': 'WU',
        'Xperi Holding Corp': 'XPER'
    }

def calculate_magic_formula_metrics(companies_df, income_df, balance_df, prices_df):
    """
    Calculate Magic Formula metrics (ROC and Earnings Yield) for all companies
    Following Joel Greenblatt's Magic Formula criteria:
    1. Filter for US market only
    2. Exclude companies with market cap < $50M
    3. Calculate proper ROC and EY using correct formulas
    """
    logging.info("Calculating Magic Formula metrics...")
    logging.info(f"Initial companies count: {len(companies_df)}")
    
    # Get latest prices and dates for each company
    latest_prices = prices_df.reset_index().groupby('Ticker').last()
    logging.info(f"Companies with price data: {len(latest_prices)}")
    
    # Calculate market cap (shares outstanding * price)
    market_cap = latest_prices['Shares Outstanding'] * latest_prices['Close']
    market_cap.name = 'Market Cap'
    logging.info(f"Companies with market cap calculated: {len(market_cap)}")
    
    # Get price dates
    price_dates = latest_prices['Date']
    price_dates.name = 'Price From'
    
    # Get most recent quarter data
    most_recent_quarters = income_df.sort_values('Report Date').groupby('SimFinId').last()
    most_recent_quarters['Most Recent Quarter Data'] = most_recent_quarters['Report Date']
    logging.info(f"Companies with recent quarter data: {len(most_recent_quarters)}")
    
    # Log date ranges
    logging.info(f"Price data range: {prices_df['Date'].min()} to {prices_df['Date'].max()}")
    logging.info(f"Income statement data range: {income_df['Report Date'].min()} to {income_df['Report Date'].max()}")
    logging.info(f"Balance sheet data range: {balance_df['Report Date'].min()} to {balance_df['Report Date'].max()}")
    
    # 1. Filter for US market only BEFORE merging
    companies_df = companies_df[companies_df['Market'].str.lower() == 'us']
    logging.info(f"After US market filter (pre-merge): {len(companies_df)}")
    
    # Merge financial data
    merged_df = pd.merge(companies_df, income_df[['SimFinId', 'Operating Income (Loss)', 'Revenue', 'Interest Expense, Net']], 
                        on='SimFinId', how='left')
    logging.info(f"After merging with income data: {len(merged_df)}")
    
    merged_df = pd.merge(merged_df, balance_df[['SimFinId', 'Total Current Assets', 'Total Current Liabilities', 
                                              'Property, Plant & Equipment, Net', 
                                              'Cash, Cash Equivalents & Short Term Investments',
                                              'Short Term Debt', 'Long Term Debt', 'Total Assets']], 
                        on='SimFinId', how='left')
    logging.info(f"After merging with balance sheet data: {len(merged_df)}")
    
    # Merge with market cap and dates
    merged_df = pd.merge(merged_df, market_cap.reset_index(), on='Ticker', how='left')
    logging.info(f"After merging with market cap data: {len(merged_df)}")
    
    merged_df = pd.merge(merged_df, price_dates.reset_index(), on='Ticker', how='left')
    merged_df = pd.merge(merged_df, most_recent_quarters[['Most Recent Quarter Data']], 
                        left_on='SimFinId', right_index=True, how='left')
    logging.info(f"After merging all data: {len(merged_df)}")
    
    # Log nulls in key columns
    key_columns = ['Operating Income (Loss)', 'Total Current Assets', 'Total Current Liabilities', 
                  'Property, Plant & Equipment, Net', 'Market Cap', 'Most Recent Quarter Data']
    for col in key_columns:
        null_count = merged_df[col].isnull().sum()
        logging.info(f"Nulls in {col}: {null_count}")
    
    # 2. Calculate proper metrics
    merged_df['EBIT'] = merged_df['Operating Income (Loss)']
    merged_df['Net Working Capital'] = merged_df['Total Current Assets'] - merged_df['Total Current Liabilities']
    merged_df['Net Fixed Assets'] = merged_df['Property, Plant & Equipment, Net']
    merged_df['Invested Capital'] = merged_df['Net Working Capital'] + merged_df['Net Fixed Assets']
    
    # Calculate Total Debt and Enterprise Value
    merged_df['Total Debt'] = merged_df['Short Term Debt'] + merged_df['Long Term Debt']
    merged_df['Enterprise Value'] = merged_df['Market Cap'] + merged_df['Total Debt'] - merged_df['Cash, Cash Equivalents & Short Term Investments']
    
    # Analyze Enterprise Value distribution
    ev_stats = merged_df['Enterprise Value'].describe()
    logging.info("\nEnterprise Value Distribution:")
    logging.info(f"Count: {ev_stats['count']}")
    logging.info(f"Mean: ${ev_stats['mean']/1e6:.2f}M")
    logging.info(f"Std: ${ev_stats['std']/1e6:.2f}M")
    logging.info(f"Min: ${ev_stats['min']/1e6:.2f}M")
    logging.info(f"25%: ${ev_stats['25%']/1e6:.2f}M")
    logging.info(f"50%: ${ev_stats['50%']/1e6:.2f}M")
    logging.info(f"75%: ${ev_stats['75%']/1e6:.2f}M")
    logging.info(f"Max: ${ev_stats['max']/1e6:.2f}M")
    
    # Log companies with negative Enterprise Value
    neg_ev_df = merged_df[merged_df['Enterprise Value'] <= 0][['Company Name', 'Ticker', 'Market Cap', 'Total Debt', 'Cash, Cash Equivalents & Short Term Investments', 'Enterprise Value']]
    neg_ev_df = neg_ev_df.head(10)  # Show first 10 examples
    logging.info("\nExample companies with negative/zero Enterprise Value:")
    for _, row in neg_ev_df.iterrows():
        logging.info(f"{row['Company Name']} ({row['Ticker']}): Market Cap=${row['Market Cap']/1e6:.2f}M, Debt=${row['Total Debt']/1e6:.2f}M, Cash=${row['Cash, Cash Equivalents & Short Term Investments']/1e6:.2f}M, EV=${row['Enterprise Value']/1e6:.2f}M")
    
    # Log metrics before filtering
    metrics = ['EBIT', 'Invested Capital', 'Enterprise Value', 'Market Cap']
    for metric in metrics:
        negative_count = len(merged_df[merged_df[metric] <= 0])
        logging.info(f"Companies with {metric} <= 0: {negative_count}")
    
    # 3. Filter out invalid entries
    filtered_df = merged_df[merged_df['Market Cap'] > 50e6]
    logging.info(f"After market cap filter: {len(filtered_df)}")
    
    filtered_df = filtered_df[filtered_df['EBIT'] > 0]
    logging.info(f"After EBIT filter: {len(filtered_df)}")
    
    filtered_df = filtered_df[filtered_df['Invested Capital'] > 0]
    logging.info(f"After Invested Capital filter: {len(filtered_df)}")
    
    filtered_df = filtered_df[filtered_df['Enterprise Value'] > 0]
    logging.info(f"After Enterprise Value filter: {len(filtered_df)}")
    
    # Filter for reasonably recent data (within last 2 years)
    two_years_ago = pd.Timestamp.now() - pd.DateOffset(years=2)
    filtered_df = filtered_df[pd.to_datetime(filtered_df['Most Recent Quarter Data']) >= two_years_ago]
    logging.info(f"After date filter (2 years): {len(filtered_df)}")
    
    # Remove duplicates based on Ticker
    filtered_df = filtered_df.drop_duplicates(subset=['Ticker'], keep='first')
    logging.info(f"After removing duplicates: {len(filtered_df)}")
    
    merged_df = filtered_df
    
    # 4. Calculate Magic Formula metrics
    # ROC = EBIT / Invested Capital
    merged_df['ROC'] = merged_df['EBIT'] / merged_df['Invested Capital']
    
    # Earnings Yield = EBIT / Enterprise Value
    merged_df['Earnings Yield'] = merged_df['EBIT'] / merged_df['Enterprise Value']

    # Calculate ranks
    merged_df['ROC Rank'] = merged_df['ROC'].rank(ascending=False)
    merged_df['EY Rank'] = merged_df['Earnings Yield'].rank(ascending=False)
    merged_df['Combined Rank'] = merged_df['ROC Rank'] + merged_df['EY Rank']

    # Sort by combined rank and take top 50
    result_df = merged_df.nsmallest(50, 'Combined Rank')

    # Format output
    result_df = result_df[['Company Name', 'Ticker', 'Market Cap', 'Price From', 'Most Recent Quarter Data']]
    result_df['Market Cap'] = result_df['Market Cap'] / 1_000_000  # Convert to millions
    result_df = result_df.sort_values('Company Name')

    logging.info(f'Final output contains {len(result_df)} companies')
    return result_df

def rank_stocks(df):
    """
    Rank stocks based on combined ROC and EY scores.
    """
    try:
        # Rank by ROC (higher is better)
        df['ROC Rank'] = df['ROC'].rank(ascending=False)
        
        # Rank by EY (higher is better)
        df['EY Rank'] = df['EY'].rank(ascending=False)
        
        # Combined rank (lower is better)
        df['Combined Rank'] = df['ROC Rank'] + df['EY Rank']
        
        # Sort by combined rank
        return df.sort_values('Combined Rank')
        
    except Exception as e:
        logger.error(f"Error ranking stocks: {str(e)}")
        raise

def main():
    """
    Main function to run the Magic Formula stock screener
    """
    setup_logging()
    logging.info("Starting Magic Formula stock screener...")
    
    try:
        # First, make ONE API call to get ALL data
        download_simfin_data()
        
        # Then load data from local files
        companies_df, income_df, balance_df, prices_df = fetch_all_stock_data()
        
        # Calculate Magic Formula metrics
        results = calculate_magic_formula_metrics(companies_df, income_df, balance_df, prices_df)
        
        # Display results
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print("\nTop 50 Magic Formula Stocks:")
        print(results.to_string(index=False))
        
        # Save to CSV
        output_file = 'magic_formula_results.csv'
        results.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()
