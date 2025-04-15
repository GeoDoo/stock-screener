import os
import pandas as pd
import numpy as np
import simfin as sf
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_simfin_api_key():
    """Get SimFin API key from environment variable."""
    api_key = os.getenv('SIMFIN_API_KEY')
    if not api_key:
        raise ValueError("SIMFIN_API_KEY environment variable not set")
    return api_key

def fetch_all_stock_data():
    """
    Fetch all US stocks data in a single API call.
    Returns a tuple of (companies_df, income_df, balance_df, prices_df)
    """
    try:
        # Set API key and initialize SimFin
        sf.set_api_key(get_simfin_api_key())
        sf.set_data_dir('simfin_data/')
        
        # Get all US companies
        logger.info("Fetching US companies data...")
        companies_df = sf.load_companies(market='us')
        logger.info(f"Companies columns: {companies_df.columns.tolist()}")
        
        # Get all SimFin IDs
        simfin_ids = companies_df['SimFinId'].tolist()
        logger.info(f"Found {len(simfin_ids)} US companies")
        
        # Fetch income statements for all companies
        logger.info("Fetching income statements...")
        income_df = sf.load_income(variant='annual', market='us')
        logger.info(f"Income columns: {income_df.columns.tolist()}")
        
        # Fetch balance sheets for all companies
        logger.info("Fetching balance sheets...")
        balance_df = sf.load_balance(variant='annual', market='us')
        logger.info(f"Balance columns: {balance_df.columns.tolist()}")
        
        # Fetch share prices
        logger.info("Fetching share prices...")
        prices_df = sf.load_shareprices(variant='daily', market='us')
        logger.info(f"Prices columns: {prices_df.columns.tolist()}")
        
        return companies_df, income_df, balance_df, prices_df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def calculate_magic_formula_metrics(companies_df, income_df, balance_df, prices_df):
    """
    Calculate Return on Capital (ROC) and Earnings Yield (EY) for all companies.
    """
    try:
        # Get the most recent year's data
        latest_year = income_df['Fiscal Year'].max()
        logger.info(f"Using data from year: {latest_year}")
        
        # Filter for latest year
        income_latest = income_df[income_df['Fiscal Year'] == latest_year]
        balance_latest = balance_df[balance_df['Fiscal Year'] == latest_year]
        
        # Get latest share prices
        # Group by SimFinId and get the last row for each company
        latest_prices = prices_df.groupby('SimFinId').last().reset_index()
        
        # Merge data
        merged_df = pd.merge(income_latest, balance_latest, 
                           on=['SimFinId', 'Fiscal Year'], 
                           suffixes=('_inc', '_bal'))
        
        # Merge with companies data to get company names
        merged_df = pd.merge(merged_df, companies_df[['SimFinId', 'Company Name']], 
                           on='SimFinId')
        
        # Merge with share prices
        merged_df = pd.merge(merged_df, latest_prices[['SimFinId', 'Close', 'Shares Outstanding']], 
                           on='SimFinId')
        
        # Calculate Market Cap using the latest shares outstanding
        merged_df['Market Cap'] = merged_df['Shares Outstanding'] * merged_df['Close']
        
        # Calculate Return on Capital (ROC)
        # ROC = EBIT / (Net Fixed Assets + Working Capital)
        merged_df['Working Capital'] = merged_df['Total Current Assets'] - merged_df['Total Current Liabilities']
        merged_df['Net Fixed Assets'] = merged_df['Property, Plant & Equipment, Net']
        merged_df['Total Capital'] = merged_df['Net Fixed Assets'] + merged_df['Working Capital']
        merged_df['ROC'] = merged_df['Operating Income (Loss)'] / merged_df['Total Capital']
        
        # Calculate Earnings Yield (EY)
        # EY = EBIT / Enterprise Value
        # Enterprise Value = Market Cap + Total Debt - Cash
        merged_df['Total Debt'] = merged_df['Long Term Debt'] + merged_df['Short Term Debt']
        merged_df['Enterprise Value'] = merged_df['Market Cap'] + merged_df['Total Debt'] - merged_df['Cash, Cash Equivalents & Short Term Investments']
        merged_df['EY'] = merged_df['Operating Income (Loss)'] / merged_df['Enterprise Value']
        
        # Filter out companies with missing or invalid data
        merged_df = merged_df[
            (merged_df['ROC'].notna()) & 
            (merged_df['EY'].notna()) & 
            (merged_df['ROC'] > 0) & 
            (merged_df['EY'] > 0) &
            (merged_df['Market Cap'] >= 50e6)  # Filter out micro-caps
        ]
        
        logger.info(f"Found {len(merged_df)} companies with valid data")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

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
    try:
        # Fetch all data
        companies_df, income_df, balance_df, prices_df = fetch_all_stock_data()
        
        # Calculate metrics
        metrics_df = calculate_magic_formula_metrics(companies_df, income_df, balance_df, prices_df)
        
        # Rank stocks
        ranked_df = rank_stocks(metrics_df)
        
        # Display top 30 stocks
        top_30 = ranked_df.head(30)[['SimFinId', 'Company Name', 'ROC', 'EY', 'Combined Rank']]
        print("\nTop 30 Magic Formula Stocks:")
        print(top_30.to_string(index=False))
        
        # Save results to CSV
        output_file = 'magic_formula_results.csv'
        ranked_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
