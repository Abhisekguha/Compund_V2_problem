# Compound V2 Wallet Credit Scoring System

This repository contains a machine learning-based credit scoring system for Compound V2 protocol wallets. The system analyzes historical transaction patterns to assign credit scores between 0-100 to each wallet, with higher scores indicating more reliable and responsible usage.

## Overview

The scoring system:
- Analyzes raw transaction data from Compound V2
- Engineers wallet-level behavioral features
- Applies a multi-factor scoring algorithm
- Produces wallet credit scores and detailed analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/compound-v2-scoring.git
cd compound-v2-scoring

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

1. Download the Compound V2 dataset from Google Drive (link in the project description)
2. Select the 3 largest files from the dataset
3. Place the files in the `data/compound_v2` directory

## Usage

```bash
# Run the main scoring script
python wallet_scorer.py
```

This will:
1. Load the transaction data
2. Preprocess and engineer features
3. Calculate credit scores for all wallets
4. Save the results to the `output` directory
5. Generate analysis of top and bottom wallets

## Output Files

The system generates the following output files:

- `all_wallet_scores.csv`: Scores for all wallets with detailed component breakdowns
- `top_1000_wallets.csv`: The 1000 highest-scoring wallets (sorted by score)
- `wallet_features.csv`: All engineered features for each wallet

## Scoring Methodology

The credit score is calculated based on 10 key components:

1. **Longevity**: Account age and activity duration
2. **Activity Frequency**: Transaction density over time
3. **Time Consistency**: Regularity of transaction patterns
4. **Repayment Behavior**: Loan repayment completeness
5. **Collateral Health**: Overcollateralization practices
6. **Liquidation History**: Penalty for liquidation events
7. **Asset Diversity**: Engagement with multiple assets
8. **Transaction Volatility**: Consistency in transaction sizes
9. **Borrow Activity**: Engagement with borrowing features
10. **Healthy Usage**: Balanced usage of protocol functions

See the [Methodology Document](methodology-document.md) for complete details on the scoring approach.

## Wallet Analysis

For insights into the characteristics of high-scoring and low-scoring wallets, see the [Wallet Analysis](wallet-analysis.md) document.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Project Structure

```
compound-v2-scoring/
├── data/
│   └── compound_v2/        # Raw data files (place downloaded files here)
├── output/                 # Generated results
├── wallet_scorer.py        # Main scoring script
├── methodology-document.md # Detailed scoring methodology
├── wallet-analysis.md      # Analysis of example wallets
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Extending the System

The scoring system can be extended in several ways:

1. **Additional Features**: Add new behavioral features to the `engineer_features` method
2. **Alternative Scoring**: Modify the `calculate_scores` method to use different scoring algorithms
3. **Time Weighting**: Add time-based weighting to prioritize recent behavior
4. **Machine Learning**: Replace the rule-based scoring with a trained model

## License

MIT