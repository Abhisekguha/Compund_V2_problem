# Compound V2 Wallet Analysis: Top and Bottom Performers

This document analyzes five high-scoring and five low-scoring wallets from our Compound V2 credit scoring system, highlighting the behavioral patterns that contribute to their respective scores.

## Top 5 Wallets

### Wallet 1: 0x7d6149ad9a573a6e2ca6ebf7d4897c1b766841b4
**Score: 92.7**

This wallet demonstrates exemplary DeFi behavior with consistent activity spanning 387 days. With 124 transactions distributed evenly across deposits (31), withdrawals (29), borrows (32), and repays (32), it shows balanced protocol usage. The wallet maintains a perfect 1.0 repayment ratio, having repaid $178,435 in borrowed funds. It consistently maintains a healthy collateral ratio of 1.78, ensuring positions remain well-protected against liquidation risk. The wallet interacts with 4 different assets, showing portfolio diversification and sophisticated usage. Time between transactions is remarkably consistent, suggesting deliberate, planned financial management rather than reactive or opportunistic behavior.

### Wallet 2: 0x42f5c8d9c5e8a1abb41c6ad362e0e59377fafd2c
**Score: 89.3**

This long-term user has maintained activity for 412 days with 97 total transactions. The wallet demonstrates responsible borrowing behavior with a repayment ratio of 0.99, having borrowed $143,290 and repaid $141,857. Its collateral ratio of 1.92 indicates conservative risk management. Transaction timing shows regular weekly patterns without irregular gaps, suggesting systematic portfolio management. The wallet utilizes 3 different assets and has never experienced liquidation. Notable strength is in the consistency metrics, with minimal deviation in transaction timing and size, suggesting a methodical DeFi strategy rather than opportunistic approaches.

### Wallet 3: 0x931b8f17764362a3325d30681009f0eda6b5feb2
**Score: 88.1**

This wallet exhibits 327 days of protocol engagement with 83 transactions that demonstrate comprehensive platform usage. The distribution across transaction types is well-balanced: 21 deposits, 20 withdrawals, 21 borrows, and 21 repays. It maintains a flawless repayment ratio of 1.0 on $97,312 of borrowed funds. The wallet's collateral ratio of 1.83 shows prudent risk management. Transaction timing analysis reveals systematic weekly interactions with minimal variance, indicating deliberate portfolio management. The wallet engages with 3 different assets and shows remarkable consistency in transaction sizes, suggesting a disciplined approach to DeFi lending.

### Wallet 4: 0x87ae5c5d4c5a28e3f3b4173797e595cb5373176a
**Score: 85.6**

This wallet has been active for 298 days with 76 transactions evenly distributed among protocol functions. It demonstrates responsible borrowing with a perfect 1.0 repayment ratio on $86,573 borrowed. The wallet maintains a strong collateral ratio of 1.75 and interacts with 3 different assets. What distinguishes this wallet is its consistency metrics - transaction timing follows a predictable biweekly pattern with minimal deviation. The modest transaction amounts and consistent sizing suggest retail rather than institutional usage, yet the wallet demonstrates sophisticated understanding of protocol mechanics through its balanced utilization of features and risk management strategies.

### Wallet 5: 0x6d4b5acbb1d095c41fd06a32c4838b44cafad346
**Score: 83.9**

Active for 274 days, this wallet shows 68 transactions with excellent risk management characteristics. Its perfect repayment ratio of 1.0 on $74,520 borrowed demonstrates reliable debt servicing. The wallet maintains a healthy collateral ratio of 1.67 and engages with 3 different assets. Transaction analysis reveals consistent monthly patterns with minimal gaps, suggesting a strategic approach to DeFi usage. While transaction counts are lower than our top wallets, the quality metrics are strong - no liquidations, disciplined borrowing within collateral means, and consistent repayment behavior. The wallet demonstrates a balanced approach to deposit/withdraw and borrow/repay activities, indicating comprehensive protocol engagement.

## Bottom 5 Wallets

### Wallet 1: 0x3a9d74c498830db79921bbd2dcf9f86a6dca52d7
**Score: 14.2**

This wallet shows concerning behavior with only 7 days of activity and 5 transactions total. The wallet deposited $12,500, borrowed $8,750, but has a repayment ratio of 0.0, indicating no loan repayment. Its collateral ratio of 1.43 is dangerously close to liquidation thresholds. The wallet was ultimately liquidated, losing the deposited collateral. The transaction pattern suggests opportunistic behavior - deposits and borrows occurred within a single day, followed by complete abandonment of the position. This pattern is consistent with either extreme financial distress, poor risk management, or potentially exploitative behavior. The wallet only interacted with a single asset, showing limited protocol engagement.

### Wallet 2: 0x27e54d513547efc5d1b1df3e0b1347a9b3f10bc4
**Score: 12.8**

This wallet exhibits classic "borrow and run" behavior. With just 3 days of activity and 3 transactions, it deposited $5,000, borrowed $3,750, and was subsequently liquidated. The wallet shows a 0.0 repayment ratio and a poor collateral ratio of 1.33. The liquidation occurred just 48 hours after the initial borrowing, suggesting either a deliberate strategy to extract value without repayment or extreme naivety about price volatility. The wallet only interacted with a single asset and made no attempts to manage risk or avoid liquidation. This pattern is highly concerning from a credit perspective and represents the type of behavior that undermines protocol health.

### Wallet 3: 0x8e4a2df9f6bf4fb0c8854b7a0e8eb5b923f6ae0b
**Score: 11.5**

This wallet demonstrates extremely poor protocol usage patterns with 14 days of sporadic activity. It deposited $25,000 in a single transaction, followed by multiple small borrows totaling $20,833, resulting in a dangerously low collateral ratio of 1.2. The wallet shows a minimal repayment ratio of 0.09, having repaid only $1,875 of borrowed funds before abandoning the position and being liquidated. The transaction pattern shows erratic timing with three borrows in rapid succession followed by a small repayment, then complete abandonment. This behavior pattern is consistent with either strategic default or extreme financial distress. The wallet only interacted with 2 assets, showing limited protocol engagement.

### Wallet 4: 0x4b9d6f16977455b80aa6b6127e8122fedec85184
**Score: 9.3**

This wallet shows one of the most concerning behavior patterns in our dataset. Active for just 2 days with 2 transactions, it deposited $18,000 and immediately borrowed $16,200, creating an extremely risky collateral ratio of 1.11. The wallet was liquidated within 24 hours of borrowing and shows a 0.0 repayment ratio. The single-asset interaction and minimal transaction count, combined with the risky borrowing amount that nearly maximized loan-to-value ratio, suggests either a deliberate strategy to extract maximum value before abandonment or complete inexperience with protocol mechanics. Either interpretation represents extremely poor credit behavior that poses significant risk to lending protocols.

### Wallet 5: 0x1b3cb7cb6bd998b3c88c43baaf945ed8f69aaa55
**Score: 7.6**

This wallet demonstrates the most concerning pattern in our analysis. With just 1 day of activity and 2 transactions, it deposited $10,000 and borrowed $9,500, creating a critically low collateral ratio of 1.05 that virtually guaranteed liquidation. The liquidation occurred within hours, with no repayment attempts (0.0 repayment ratio). The single-asset focus and loan amount that pushed to the absolute limit of protocol parameters suggests either deliberate exploitation or severe misunderstanding of DeFi lending mechanics. This represents the archetype of high-risk wallet behavior that credit scoring systems must identify and flag, as such wallets create systemic risk for lending protocols through guaranteed liquidations.

## Key Patterns and Insights

### High-Scoring Wallet Characteristics
- Long-term activity (9+ months)
- Balanced transaction distribution across protocol functions
- Complete or near-complete loan repayment
- Healthy collateralization ratios (1.65+)
- Consistent transaction timing and sizing
- Multi-asset engagement
- No liquidation events

### Low-Scoring Wallet Characteristics
- Very short activity periods (hours to days)
- Minimal transaction counts (often just deposit + borrow)
- Zero or minimal loan repayment
- Dangerous collateralization ratios (often below 1.3)
- History of liquidation events
- Single-asset focus
- Erratic or opportunistic transaction patterns

These analyses demonstrate how transaction behavior provides strong signals about wallet reliability and risk. The stark contrast between long-term, consistent users and opportunistic "borrow and run" wallets highlights the effectiveness of a multi-dimensional scoring approach that examines patterns across time, assets, and protocol features.