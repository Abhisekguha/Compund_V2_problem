# Compound V2 Wallet Credit Scoring System

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
from tqdm import tqdm  # Import tqdm for progress bars
import time

warnings.filterwarnings('ignore')

# Global constants
MIN_SCORE = 0
MAX_SCORE = 100

class CompoundWalletScorer:
    """
    A class to score Compound V2 wallets based on their transaction behavior
    """
    
    def __init__(self, data_dir):
        """
        Initialize the scorer with the directory containing Compound V2 data files
        
        Args:
            data_dir (str): Path to directory containing JSON data files
        """
        self.data_dir = data_dir
        self.raw_data = None
        self.wallet_features = None
        self.wallet_scores = None
        
    def load_data(self, top_n_files=3):
        """
        Load JSON data from the specified directory
        
        Args:
            top_n_files (int): Number of largest files to load
        """
        print(f"Loading data from {self.data_dir}")
        
        # Get list of JSON files and their sizes
        files = [(f, os.path.getsize(os.path.join(self.data_dir, f))) 
                for f in os.listdir(self.data_dir) 
                if f.endswith('.json')]
        
        # Sort by size (largest first) and take top N
        files.sort(key=lambda x: x[1], reverse=True)
        selected_files = [f[0] for f in files[:top_n_files]]
        
        print(f"Selected files: {selected_files}")
        
        # Dictionary to store combined data
        combined_data = {
            "deposits": [],
            "withdraws": [],
            "borrows": [],
            "repays": [],
            "liquidates": []
        }
        
        # Setup progress bar for file loading
        file_pbar = tqdm(selected_files, desc="Loading files", unit="file")
        
        # Load and combine data from each file
        for file_name in file_pbar:
            file_path = os.path.join(self.data_dir, file_name)
            file_pbar.set_description(f"Processing {file_name}")
            
            start_time = time.time()
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                for action_type in combined_data.keys():
                    if action_type in data:
                        combined_data[action_type].extend(data[action_type])
            
            elapsed = time.time() - start_time
            file_pbar.set_postfix(elapsed=f"{elapsed:.2f}s")
        
        self.raw_data = combined_data
        print(f"Loaded {len(combined_data['deposits'])} deposits, {len(combined_data['withdraws'])} withdrawals, "
              f"{len(combined_data['borrows'])} borrows, {len(combined_data['repays'])} repays, "
              f"{len(combined_data['liquidates'])} liquidations")
    
    def preprocess_data(self):
        """
        Convert raw JSON data into pandas DataFrames and perform preprocessing
        """
        if self.raw_data is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")
            
        print("Preprocessing data...")
        start_time = time.time()
        
        # Convert each action type to a DataFrame
        action_dfs = {}
        preprocess_pbar = tqdm(self.raw_data.items(), desc="Preprocessing action types", unit="type")
        
        for action_type, data in preprocess_pbar:
            if data:  # Only process if there's data
                preprocess_pbar.set_description(f"Processing {action_type}")
                df = pd.json_normalize(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                
                # Extract wallet address
                if 'account.id' in df.columns:
                    df['wallet'] = df['account.id']
                
                # Convert amounts to numeric values
                for col in ['amount', 'amountUSD']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Add action type as a column
                df['action_type'] = action_type
                
                # Add asset symbol if available
                if 'asset.symbol' in df.columns:
                    df['asset'] = df['asset.symbol']
                
                action_dfs[action_type] = df
        
        # Store processed DataFrames
        self.action_dfs = action_dfs
        
        # Create a combined timeline of all events
        print("Combining data into a timeline...")
        dfs_to_combine = []
        for action_type, df in action_dfs.items():
            # Select common columns for timeline
            cols_to_use = ['wallet', 'timestamp', 'amount', 'amountUSD', 'action_type', 'asset']
            cols_to_use = [col for col in cols_to_use if col in df.columns]
            dfs_to_combine.append(df[cols_to_use])
        
        self.timeline_df = pd.concat(dfs_to_combine, ignore_index=True)
        self.timeline_df.sort_values('timestamp', inplace=True)
        
        elapsed = time.time() - start_time
        print(f"Preprocessed data for {len(self.timeline_df['wallet'].unique())} unique wallets in {elapsed:.2f} seconds")
    
    def engineer_features(self):
        """
        Engineer wallet-level features from the processed data
        """
        if not hasattr(self, 'timeline_df'):
            raise ValueError("Data has not been preprocessed. Call preprocess_data() first.")
            
        print("Engineering features...")
        start_time = time.time()
        
        # Get unique wallets
        wallets = self.timeline_df['wallet'].unique()
        
        # Initialize dictionary to store features for each wallet
        wallet_features = {}
        
        # Process each wallet with a progress bar
        wallet_pbar = tqdm(wallets, desc="Processing wallets", unit="wallet")
        
        for wallet in wallet_pbar:
            # Update progress bar description every 100 wallets to avoid slowdown
            if wallet_pbar.n % 100 == 0:
                wallet_pbar.set_description(f"Processing wallet {wallet_pbar.n}/{len(wallets)}")
            
            # Filter for this wallet's activity
            wallet_df = self.timeline_df[self.timeline_df['wallet'] == wallet]
            
            # Basic activity metrics
            features = {
                'wallet': wallet,
                'first_activity': wallet_df['timestamp'].min(),
                'last_activity': wallet_df['timestamp'].max(),
                'total_transactions': len(wallet_df),
            }
            
            # Calculate age in days
            features['account_age_days'] = (features['last_activity'] - features['first_activity']).total_seconds() / (60*60*24)
            
            # Frequency metrics
            if features['account_age_days'] > 0:
                features['transactions_per_day'] = features['total_transactions'] / features['account_age_days']
            else:
                features['transactions_per_day'] = features['total_transactions']  # All in one day
            
            # Action type counts and proportions
            action_counts = wallet_df['action_type'].value_counts()
            total_actions = len(wallet_df)
            
            for action in ['deposits', 'withdraws', 'borrows', 'repays', 'liquidates']:
                count = action_counts.get(action, 0)
                features[f'{action}_count'] = count
                features[f'{action}_proportion'] = count / total_actions if total_actions > 0 else 0
            
            # Liquidation risk metrics
            features['has_been_liquidated'] = 1 if features['liquidates_count'] > 0 else 0
            
            # Deposit and withdrawal metrics
            deposits = wallet_df[wallet_df['action_type'] == 'deposits']
            withdraws = wallet_df[wallet_df['action_type'] == 'withdraws']
            
            # Calculate total deposit and withdrawal amounts
            features['total_deposit_usd'] = deposits['amountUSD'].sum() if not deposits.empty else 0
            features['total_withdraw_usd'] = withdraws['amountUSD'].sum() if not withdraws.empty else 0
            
            # Calculate net position
            features['net_position_usd'] = features['total_deposit_usd'] - features['total_withdraw_usd']
            
            # Borrow and repay metrics
            borrows = wallet_df[wallet_df['action_type'] == 'borrows']
            repays = wallet_df[wallet_df['action_type'] == 'repays']
            
            # Calculate borrow and repay amounts
            features['total_borrow_usd'] = borrows['amountUSD'].sum() if not borrows.empty else 0
            features['total_repay_usd'] = repays['amountUSD'].sum() if not repays.empty else 0
            
            # Calculate repayment ratio
            if features['total_borrow_usd'] > 0:
                features['repayment_ratio'] = features['total_repay_usd'] / features['total_borrow_usd']
            else:
                features['repayment_ratio'] = 1.0  # Default to 1.0 for wallets with no borrows
            
            # Time consistency metrics
            if len(wallet_df) > 1:
                timestamps = wallet_df['timestamp'].sort_values()
                time_diffs = timestamps.diff().dropna()
                features['mean_time_between_txs_hours'] = time_diffs.mean().total_seconds() / 3600
                features['std_time_between_txs_hours'] = time_diffs.std().total_seconds() / 3600 if len(time_diffs) > 1 else 0
                features['max_time_gap_days'] = time_diffs.max().total_seconds() / (24*3600)
            else:
                features['mean_time_between_txs_hours'] = 0
                features['std_time_between_txs_hours'] = 0
                features['max_time_gap_days'] = 0
            
            # Asset diversity
            features['unique_assets'] = wallet_df['asset'].nunique()
            
            # Transaction size variability
            if 'amountUSD' in wallet_df.columns:
                features['mean_tx_size_usd'] = wallet_df['amountUSD'].mean()
                features['max_tx_size_usd'] = wallet_df['amountUSD'].max()
                features['min_tx_size_usd'] = wallet_df['amountUSD'].min()
                features['std_tx_size_usd'] = wallet_df['amountUSD'].std() if len(wallet_df) > 1 else 0
                features['tx_size_variation'] = features['std_tx_size_usd'] / features['mean_tx_size_usd'] if features['mean_tx_size_usd'] > 0 else 0
            
            # Deposit to borrow ratio (overcollateralization)
            if features['total_borrow_usd'] > 0:
                features['collateral_ratio'] = features['total_deposit_usd'] / features['total_borrow_usd']
            else:
                features['collateral_ratio'] = float('inf')  # infinite collateral for no borrows
            
            # Store features for this wallet
            wallet_features[wallet] = features
            
            # Update progress bar stats occasionally
            if wallet_pbar.n % 500 == 0:
                wallet_pbar.set_postfix(elapsed=f"{time.time() - start_time:.1f}s")
        
        # Convert to DataFrame
        self.wallet_features = pd.DataFrame(list(wallet_features.values()))
        elapsed = time.time() - start_time
        print(f"Engineered {len(self.wallet_features.columns)} features for {len(self.wallet_features)} wallets in {elapsed:.2f} seconds")
    
    def calculate_scores(self):
        """
        Calculate credit scores for wallets based on engineered features
        """
        if self.wallet_features is None:
            raise ValueError("Features have not been engineered. Call engineer_features() first.")
            
        print("Calculating wallet scores...")
        start_time = time.time()
        
        # Copy the feature DataFrame to avoid modifying the original
        df = self.wallet_features.copy()
        
        # Filter out wallets with very limited activity (e.g., single transaction)
        df = df[df['total_transactions'] > 1].copy()
        
        # Define key scoring metrics
        score_components = {
            # Activity Consistency (higher is better)
            'longevity': lambda x: min(x['account_age_days'] / 365, 2),  # Cap at 2 years
            'activity_frequency': lambda x: min(x['transactions_per_day'], 5) / 5,  # Normalize to [0,1]
            'time_consistency': lambda x: 1 - min(x['std_time_between_txs_hours'] / 168, 1),  # Lower variation is better
            
            # Repayment Behavior (higher is better)
            'repayment_ratio': lambda x: min(x['repayment_ratio'], 1),  # Cap at 1
            'collateral_health': lambda x: min(x['collateral_ratio'], 2) / 2,  # Normalize to [0,1]
            
            # Liquidation History (lower is better)
            'liquidation_penalty': lambda x: -x['has_been_liquidated'] * 0.5,  # 50% penalty for liquidation
            
            # Transaction Pattern (diversity is good, extreme volatility is bad)
            'asset_diversity': lambda x: min(x['unique_assets'] / 3, 1),  # Normalize to [0,1]
            'tx_size_volatility': lambda x: -min(x['tx_size_variation'] / 10, 1),  # Penalize high volatility
            
            # Utilization Patterns
            'borrow_activity': lambda x: 0.5 if x['borrows_count'] > 0 else 0,  # Bonus for using borrows
            'healthy_usage': lambda x: 0.5 if (x['deposits_count'] > 0 and 
                                              x['withdraws_count'] > 0 and 
                                              x['borrows_count'] > 0 and 
                                              x['repays_count'] > 0) else 0  # Bonus for using all features
        }
        
        # Calculate the raw score for each component with progress bar
        score_pbar = tqdm(score_components.items(), desc="Computing score components", unit="component")
        for component, score_func in score_pbar:
            score_pbar.set_description(f"Computing {component}")
            df[f'score_{component}'] = df.apply(score_func, axis=1)
        
        # Calculate overall raw score (sum of all components)
        score_columns = [f'score_{component}' for component in score_components.keys()]
        df['raw_score'] = df[score_columns].sum(axis=1)
        
        # Normalize the raw score to 0-100
        min_possible_score = -0.5  # The minimum possible score (with liquidation penalty)
        max_possible_score = sum([1, 1, 1, 1, 1, 0, 1, 0, 0.5, 0.5])  # Max possible from all positive components
        
        # Apply normalization
        df['normalized_score'] = ((df['raw_score'] - min_possible_score) / 
                                 (max_possible_score - min_possible_score)) * 100
        
        # Ensure scores are within bounds
        df['final_score'] = np.clip(df['normalized_score'], MIN_SCORE, MAX_SCORE)
        
        # Store the results
        self.wallet_scores = df[['wallet', 'final_score'] + score_columns + ['raw_score', 'normalized_score']]
        
        elapsed = time.time() - start_time
        print(f"Calculated scores for {len(self.wallet_scores)} wallets in {elapsed:.2f} seconds")
    
    def get_top_wallets(self, n=1000):
        """
        Get the top N wallets by score
        
        Args:
            n (int): Number of top wallets to return
            
        Returns:
            DataFrame: Top N wallets with their scores
        """
        if self.wallet_scores is None:
            raise ValueError("Scores have not been calculated. Call calculate_scores() first.")
            
        # Sort wallets by score (descending) and take top N
        top_wallets = self.wallet_scores.sort_values('final_score', ascending=False).head(n)
        return top_wallets
    
    def save_results(self, output_dir):
        """
        Save the results to files
        
        Args:
            output_dir (str): Directory to save the results
        """
        if self.wallet_scores is None:
            raise ValueError("Scores have not been calculated. Call calculate_scores() first.")
        
        print("Saving results...")
        start_time = time.time()
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all wallet scores
        all_scores_path = os.path.join(output_dir, 'all_wallet_scores.csv')
        self.wallet_scores.to_csv(all_scores_path, index=False)
        print(f"Saved all wallet scores to {all_scores_path}")
        
        # Save top 1000 wallets
        top_wallets = self.get_top_wallets(1000)
        top_wallets_path = os.path.join(output_dir, 'top_1000_wallets.csv')
        top_wallets.to_csv(top_wallets_path, index=False)
        print(f"Saved top 1000 wallets to {top_wallets_path}")
        
        # Save features for analysis
        features_path = os.path.join(output_dir, 'wallet_features.csv')
        self.wallet_features.to_csv(features_path, index=False)
        print(f"Saved wallet features to {features_path}")
        
        elapsed = time.time() - start_time
        print(f"Results saved in {elapsed:.2f} seconds")
    
    def analyze_top_and_bottom_wallets(self, n=5):
        """
        Analyze the characteristics of top and bottom scoring wallets
        
        Args:
            n (int): Number of wallets to analyze from each end
            
        Returns:
            dict: Analysis results
        """
        if self.wallet_scores is None or self.wallet_features is None:
            raise ValueError("Scores and features must be calculated first.")
        
        print(f"Analyzing top and bottom {n} wallets...")
        start_time = time.time()
        
        # Merge scores and features
        analysis_df = pd.merge(self.wallet_scores, self.wallet_features, on='wallet')
        
        # Get top and bottom N wallets
        top_wallets = analysis_df.nlargest(n, 'final_score')
        bottom_wallets = analysis_df.nsmallest(n, 'final_score')
        
        # Prepare analysis results
        analysis = {
            'top_wallets': self._analyze_wallet_group(top_wallets, "Top"),
            'bottom_wallets': self._analyze_wallet_group(bottom_wallets, "Bottom")
        }
        
        elapsed = time.time() - start_time
        print(f"Analysis completed in {elapsed:.2f} seconds")
        
        return analysis
    
    def _analyze_wallet_group(self, wallet_group, group_name):
        """
        Helper method to analyze a group of wallets
        
        Args:
            wallet_group (DataFrame): Group of wallets to analyze
            group_name (str): Name of the group (e.g., "Top" or "Bottom")
            
        Returns:
            list: Analysis for each wallet in the group
        """
        wallet_analyses = []
        
        for _, wallet in wallet_group.iterrows():
            analysis = {
                'wallet_address': wallet['wallet'],
                'score': wallet['final_score'],
                'key_metrics': {
                    'transactions': wallet['total_transactions'],
                    'account_age_days': wallet['account_age_days'],
                    'deposits_count': wallet['deposits_count'],
                    'withdraws_count': wallet['withdraws_count'],
                    'borrows_count': wallet['borrows_count'],
                    'repays_count': wallet['repays_count'],
                    'liquidates_count': wallet['liquidates_count'],
                    'total_deposit_usd': wallet['total_deposit_usd'],
                    'total_withdraw_usd': wallet['total_withdraw_usd'],
                    'total_borrow_usd': wallet['total_borrow_usd'],
                    'total_repay_usd': wallet['total_repay_usd'],
                    'repayment_ratio': wallet['repayment_ratio'],
                    'collateral_ratio': wallet['collateral_ratio'],
                    'unique_assets': wallet['unique_assets']
                },
                'score_components': {
                    component.replace('score_', ''): wallet[component] 
                    for component in wallet.index if component.startswith('score_')
                },
                'analysis': self._generate_wallet_narrative(wallet, group_name)
            }
            wallet_analyses.append(analysis)
        
        return wallet_analyses
    
    def _generate_wallet_narrative(self, wallet, group_name):
        """
        Generate a narrative analysis for a wallet
        
        Args:
            wallet (Series): Wallet data
            group_name (str): Group name (e.g., "Top" or "Bottom")
            
        Returns:
            str: Narrative analysis
        """
        # Start with group-specific template
        if group_name == "Top":
            narrative = f"This is a high-scoring wallet with a score of {wallet['final_score']:.2f}. "
        else:
            narrative = f"This is a low-scoring wallet with a score of {wallet['final_score']:.2f}. "
        
        # Account activity
        if wallet['account_age_days'] > 180:
            narrative += f"It has maintained activity for {wallet['account_age_days']:.0f} days, "
        else:
            narrative += f"It has a relatively short history of {wallet['account_age_days']:.0f} days, "
        
        narrative += f"with {wallet['total_transactions']} total transactions. "
        
        # Transaction patterns
        if wallet['deposits_count'] > 0 and wallet['withdraws_count'] > 0:
            narrative += f"The wallet has made {wallet['deposits_count']} deposits and {wallet['withdraws_count']} withdrawals. "
        
        # Borrowing behavior
        if wallet['borrows_count'] > 0:
            narrative += f"It has borrowed {wallet['borrows_count']} times for a total of ${wallet['total_borrow_usd']:.2f} USD. "
            
            if wallet['repayment_ratio'] >= 0.95:
                narrative += "It has fully repaid its loans, "
            elif wallet['repayment_ratio'] >= 0.5:
                narrative += f"It has partially repaid its loans (ratio: {wallet['repayment_ratio']:.2f}), "
            else:
                narrative += f"It has repaid only a small portion of its loans (ratio: {wallet['repayment_ratio']:.2f}), "
        else:
            narrative += "This wallet has never borrowed. "
        
        # Liquidation history
        if wallet['has_been_liquidated'] == 1:
            narrative += "This wallet has been liquidated, which significantly impacts its score. "
        
        # Asset diversity
        if wallet['unique_assets'] > 2:
            narrative += f"It shows diversity by interacting with {wallet['unique_assets']} different assets. "
        
        # Activity consistency
        if wallet['std_time_between_txs_hours'] < 24:
            narrative += "The wallet shows consistent activity patterns. "
        elif wallet['max_time_gap_days'] > 30:
            narrative += f"There are long gaps in activity (max: {wallet['max_time_gap_days']:.0f} days). "
        
        # Collateral health
        if wallet['collateral_ratio'] > 1.5 and wallet['borrows_count'] > 0:
            narrative += "It maintains healthy collateralization levels. "
        elif wallet['borrows_count'] > 0:
            narrative += f"Its collateralization ratio ({wallet['collateral_ratio']:.2f}) is concerning. "
        
        return narrative

# Main execution
def main():
    # Track overall execution time
    total_start_time = time.time()
    
    # Set the paths
    data_dir = r"C:\Users\ADMIN\Desktop\compund-v2\data\compound_v2"  
    output_dir = r"C:\Users\ADMIN\Desktop\compund-v2\output"
    
    # Initialize the scorer
    scorer = CompoundWalletScorer(data_dir)
    
    # Run the scoring pipeline
    scorer.load_data(top_n_files=3)
    scorer.preprocess_data()
    scorer.engineer_features()
    scorer.calculate_scores()
    
    # Save the results
    scorer.save_results(output_dir)
    
    # Analyze top and bottom wallets
    analysis = scorer.analyze_top_and_bottom_wallets(n=5)
    
    # Print sample analysis
    print("\nTop 5 Wallet Analysis:")
    for i, wallet_analysis in enumerate(analysis['top_wallets']):
        print(f"\n{i+1}. Wallet: {wallet_analysis['wallet_address']}")
        print(f"   Score: {wallet_analysis['score']:.2f}")
        print(f"   Analysis: {wallet_analysis['analysis']}")
    
    print("\nBottom 5 Wallet Analysis:")
    for i, wallet_analysis in enumerate(analysis['bottom_wallets']):
        print(f"\n{i+1}. Wallet: {wallet_analysis['wallet_address']}")
        print(f"   Score: {wallet_analysis['score']:.2f}")
        print(f"   Analysis: {wallet_analysis['analysis']}")
    
    # Print total execution time
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    main()