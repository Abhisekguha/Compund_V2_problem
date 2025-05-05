# Compound V2 Wallet Credit Scoring Methodology

## Overview

This document outlines the methodology developed for creating a credit scoring system for wallets interacting with the Compound V2 protocol. The scoring system assigns values between 0 and 100 to each wallet based on historical transaction patterns, with higher scores indicating more reliable and responsible behavior.

## Core Principles of the Scoring System

The scoring system is built around four key categories that define "good" and "bad" wallet behavior in DeFi lending:

1. **Activity Longevity and Consistency**
2. **Loan Repayment Behavior**
3. **Risk Management**
4. **Platform Engagement Patterns**

## Feature Engineering

The features engineered from raw transaction data include:

### Activity Metrics
- Account age (days since first transaction)
- Total number of transactions
- Transaction frequency (transactions per day)
- Time consistency between transactions
- Activity gaps (maximum time between consecutive transactions)

### Financial Behavior Metrics
- Deposit and withdrawal patterns
- Total deposited and withdrawn amounts
- Net position (deposits minus withdrawals)
- Borrow and repay activity
- Repayment ratio (repaid amount / borrowed amount)
- Collateralization ratio (deposits / borrows)

### Risk Metrics
- Liquidation history
- Transaction size variability
- Asset diversity

## Scoring Algorithm Components

The final score is calculated using a multi-component system:

### 1. Longevity Score (0-1)
- Rewards wallets that have been active for longer periods
- Capped at 2 years (with max score at that point)

### 2. Activity Frequency Score (0-1)
- Measures how regularly the wallet interacts with the protocol
- Normalized transaction frequency capped at 5 transactions per day

### 3. Time Consistency Score (0-1)
- Rewards consistent transaction patterns
- Penalizes highly irregular activity that might indicate opportunistic behavior

### 4. Repayment Behavior Score (0-1)
- Measures how completely a wallet repays its loans
- Perfect score for full repayment, scaled down for partial repayment

### 5. Collateral Health Score (0-1)
- Measures overcollateralization practices
- Rewards maintaining healthy collateral ratios

### 6. Liquidation Penalty (-0.5)
- Significant negative impact for wallets that have been liquidated

### 7. Asset Diversity Score (0-1)
- Rewards interaction with multiple assets
- Indicates more sophisticated and legitimate usage

### 8. Transaction Size Volatility (-1 to 0)
- Penalizes extreme variations in transaction sizes
- Extremely volatile transaction sizes may indicate abnormal patterns

### 9. Borrow Activity Score (0 or 0.5)
- Bonus for wallets that utilize borrowing functionality
- Rewards engagement with core protocol features

### 10. Healthy Usage Score (0 or 0.5)
- Bonus for wallets that use all major protocol functions
- Indicates comprehensive engagement with the platform

## Score Normalization

The raw score is calculated by summing all component scores, then normalized to a 0-100 scale:

```
normalized_score = ((raw_score - min_possible_score) / (max_possible_score - min_possible_score)) * 100
```

Where:
- `min_possible_score` = -0.5 (with liquidation penalty)
- `max_possible_score` = 7.0 (sum of maximum values from all positive components)

The final score is clipped to ensure it falls within the 0-100 range.

## Defining Good vs. Bad Behavior

### "Good" Wallet Behavior
- Consistent, long-term activity
- Full repayment of borrowed funds
- Healthy collateralization ratios
- Diverse asset usage
- No liquidation events
- Regular, predictable transaction patterns
- Engagement with multiple protocol features

### "Bad" Wallet Behavior
- Irregular, short-term activity
- Low or no loan repayment
- Poor collateralization practices
- History of liquidations
- Extreme transaction size volatility
- Single-asset focus
- Limited feature engagement

## Interpretation of Scores

- **80-100**: Excellent credit behavior - consistent, responsible usage with strong repayment history
- **60-79**: Good credit behavior - generally reliable with minor inconsistencies
- **40-59**: Moderate credit behavior - some concerning patterns but also positive indicators
- **20-39**: Poor credit behavior - significant issues with repayment or risk management
- **0-19**: Very poor credit behavior - multiple red flags, likely liquidated or abandoned positions

## Evaluation and Future Improvements

The scoring system is designed to be:
- Explainable: Each component has a clear business rationale
- Balanced: Considers both positive and negative behaviors
- Robust: Handles various wallet types and usage patterns

Future improvements could include:
- Time-weighted scoring to emphasize recent behavior
- Market condition adjustments
- Network analysis to detect related wallets
- ML-based anomaly detection for more sophisticated pattern recognition