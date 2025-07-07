# Risk Management Features

This document describes the risk management features implemented in the Deriv Trading Bot.

## Overview

The trading bot includes several risk management features to help protect your account from excessive losses. These features are designed to automatically stop trading when certain risk thresholds are reached.

## Available Risk Limits

### 1. Maximum Trades Per Session

**Default: 20 trades**

Limits the number of trades that can be executed in a single trading session. When this limit is reached, the bot will stop placing new trades until you reset the session or adjust the limit.

### 2. Maximum Consecutive Losses

**Default: 5 trades**

Stops trading after a specified number of consecutive losing trades. This helps prevent continued trading during unfavorable market conditions.

### 3. Maximum Daily Loss

**Default: $0 (disabled)**

Sets a maximum amount you're willing to lose in a single day. When losses reach this threshold, trading stops for the day. Set to 0 to disable this limit.

### 4. Cooling Period

**Default: 0 minutes (disabled)**

When a risk limit is hit, the bot can enter a "cooling period" where trading is suspended for a specified number of minutes. This provides time for market conditions to change or for you to review your strategy before trading resumes.

## Using Risk Management Features

### Setting Risk Limits

You can adjust risk limits through the bot's interface or by modifying the settings directly:

```python
# Example
bot.set_risk_limits(
    max_trades=30,       # Maximum 30 trades per session
    max_losses=3,        # Stop after 3 consecutive losses
    max_daily_loss=50,   # Stop if daily losses exceed $50
    cooling_period=15    # Wait 15 minutes after hitting limits
)
```

### Resetting Risk Counters

To reset the risk management counters after hitting a limit:

```python
# Reset session-specific counters
bot.reset_risk_management()

# Full reset (including daily loss counter)
bot.reset_risk_management(full_reset=True)
```

## Recommended Settings

The optimal risk settings depend on your trading strategy, account size, and risk tolerance. Here are some general recommendations:

- **Conservative**: Max 10 trades, 3 consecutive losses, $20 daily loss, 30 min cooling
- **Moderate**: Max 20 trades, 5 consecutive losses, $50 daily loss, 15 min cooling
- **Aggressive**: Max 50 trades, 8 consecutive losses, $100 daily loss, 5 min cooling

## Monitoring Risk Status

You can monitor the current risk status using the bot's status display:

```python
status = bot.get_bot_status()
risk_info = status['risk_management']

print(f"Trades: {risk_info['current_trades']}/{risk_info['max_trades_per_session']}")
print(f"Consecutive losses: {risk_info['current_consecutive_losses']}")
print(f"Daily loss: ${risk_info['current_daily_loss']}")
```

## Best Practices

1. **Start Conservative**: Begin with stricter limits and adjust as you gain confidence
2. **Daily Reset**: Perform a full reset at the start of each trading day
3. **Adjust to Market**: Use stricter limits in volatile markets
4. **Review After Limits**: When limits are hit, review your strategy before continuing
