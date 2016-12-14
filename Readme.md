
## [OpenAI Gym](https://gym.openai.com/) Environment for Trading

### Environment for reinforcement-learning algorithmic trading models

The Trading Environment provides an environment for single-instrument trading
using historical bar data.

### Actions

The trading model is simple.  On each step, you have the opportunity
to allocate from [-1,1] of your cash.  -1 means that you've allocated
all of your money to a short position.  1 means you've allocated all
your cash to a long position.  0 means you are flat.  When you trade,
you incur a cost for trading.  When you don't you, incur a smaller
cost (so that not trading isn't really viable).


### Key Environment Parameters:

1. TradingCost - Cost in basis-points of trading.  Default Value: 10 bps.

2. TimeDecayCost - Cost of time in basis-points (imagine you're in a negative
interest rate environment).  Default value: 1 bps.

...