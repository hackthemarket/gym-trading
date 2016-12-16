import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter

import quandl
import numpy as np
from numpy import random
import pandas as pd
import logging
import pdb

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)

EpisodeDays = 252 # trading records in episode

LoserNAV = 0  # if nav drops below this, you lose
WinnerNAV = 2 # if nav climbs above this, you win

class QuandlEnvSrc(object):
  ''' 
  Quandl-based implementation of a TradingEnv's data source.
  
  Pulls data from Quandl, preps for use by TradingEnv and then 
  acts as data provider for each new episode.
  '''

  MinPercentileDays = 100 
  QuandlAuthToken = ""
  Name = "GOOG/NYSE_SPY" #"GOOG/NYSE_IBM"

  def __init__(self, name=Name, auth=QuandlAuthToken, scale=True):
    self.name = name
    self.auth = auth
    log.info('getting data for %s from quandl...',QuandlEnvSrc.Name)
    df = quandl.get(self.name) if self.auth=='' else quandl.get(self.name, authtoken=self.auth)
    log.info('got data for %s from quandl...',QuandlEnvSrc.Name)
    
    df = df[ ~np.isnan(df.Volume)][['Close','Volume']]
    # we calculate returns and percentiles, then kill nans
    df = df[['Close','Volume']]   
    df.Volume.replace(0,1,inplace=True) # days shouldn't have zero volume..
    df['Return'] = (df.Close-df.Close.shift())/df.Close.shift()
    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    df['ClosePctl'] = df.Close.expanding(self.MinPercentileDays).apply(pctrank)
    df['VolumePctl'] = df.Volume.expanding(self.MinPercentileDays).apply(pctrank)
    df.dropna(axis=0,inplace=True)
    R = df.Return
    if scale:
      mean_values = df.mean(axis=0)
      std_values = df.std(axis=0)
      df = (df - np.array(mean_values))/ np.array(std_values)
    df['Return'] = R # we don't want our returns scaled
    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.count = 0
    
  def reset(self):
    # we want contiguous data
    self.idx = np.random.randint( low = 0, high=len(self.data.index)-EpisodeDays )
    self.count = 0
    #self.max = self.idx + EpisodeDays

  def step(self):    
    obs = self.data.iloc[self.idx].as_matrix()
    self.idx += 1
    self.count += 1
    #print self.count
    done = self.count >= EpisodeDays 
    return obs,done
    
class TradingEnv(gym.Env):
  """This gym implements a very simple trading environment for reinforcement learning.

  The gym provides daily observations based on real market data pulled
  from Quandl on, by default, the SPY etf. An episode is defined as 252
  contiguous days sampled from the overall dataset. Each day is one
  'step' within the gym and for each step, the algo has a choice:

  SHORT (0)
  FLAT (1)
  LONG (2)

  If you trade, you will be charged, by default, 10 BPS of the size of
  your trade. Thus, going from short to long costs twice as much as
  going from short to/from flat. Not trading also has a default cost of
  1 BPS per step. Nobody said it would be easy!

  At the beginning of your episode, you are allocated 1 unit of
  cash. This is your starting Net Asset Value (NAV). If your NAV drops
  to 0, your episode is over and you lose. If your NAV hits 2.0, then
  you win.

  At this point, I'm still figuring out what an appropriate condition
  for 'solved' would be in the context of OpenAI's 'Universe'...
  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.src = QuandlEnvSrc() 
    self.trading_cost_bps = 1e-3
    self.time_cost_bps    = 1e-4
    self.action_space = spaces.Discrete( 3 )
    self.observation_space= spaces.Box( self.src.min_values,
                                        self.src.max_values)
    self.reset()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    observation, done = self.src.step()
    # Close    Volume     Return  ClosePctl  VolumePctl
    yret = observation[2]
    p = action - 1     
    trade = p - self.position 
    trade_costs = self.nav * abs(trade) * self.trading_cost_bps 
    costs = trade_costs + (self.nav * self.time_cost_bps)
    daypnl = self.nav * ((self.position * yret) - costs )
    reward = daypnl
    self.nav = self.nav + daypnl
    self.position = p
    if self.nav <= LoserNAV:
      done = True
      reward = -1
    if self.nav >= WinnerNAV:
      done = True
      reward = 1
      
    info = { 'pnl': daypnl, 'nav':self.nav, 'costs':costs }

    return observation, reward, done, info
  
  def _reset(self):
    self.position = 0.0
    self.nav = 1.0
    self.src.reset()
    return self.src.step()[0]
    
  def _render(self, mode='human', close=False):
    #... TODO
    pass
  
class DayRunnr(object):
    """Simple object to aggregate stats on batch runs of 
    day-frequency trading strategies.
    
    We expect a compliant strategy to accept an observation and environment
    and return an action:
    
    Action a = strategy( observation, environment )
    """
    
    def __init__(self, env):
        self._env = env

    def _sharpe(self, Returns, freq=252) :
        """Given a set of returns, calculates naive (rfr=0) sharpe """
        return (np.sqrt(freq) * np.mean(Returns))/np.std(Returns)
    
    def run_strats( self, strategy, episodes=1 ):
        """ run provided strategy the specified # of times, returning a dataframe
        summarizing activity """
        ars = np.zeros(episodes)        # annualized return of each strategy
        sharpes = np.zeros(episodes)    # sharpe of each strategy
        costs = np.zeros(episodes)  # total costs of each strategy
        actcnt = Counter()
        for i in range(episodes):
            df,_ = self.run_strat(strategy, actcnt)
            nl = df.nav.shift().fillna(0)
            R = ((df.nav - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
            sharpes[i] = self._sharpe(R)
            ars[i] = df.nav.iloc[-1] - df.nav.iloc[0]
            costs[i] = df.costs.sum()
        df = pd.DataFrame({'ann_ret':ars,'sharpe':sharpes,'costs':costs},
                         columns=['ann_ret','sharpe','costs'])
        return df,actcnt
        
    def run_strat(self,  strategy, actcnt=None):
        """run provided strategy, returns dataframe with all steps and 
        a counter of the actions which can be passed-in """
        if actcnt is None:
            actcnt = Counter()
        observation = self._env.reset()
        days = 252
        navs = np.ones(days)
        costs = np.zeros(days)
        pnls = np.zeros(days)
        actions = np.zeros(days)
        rewards = np.zeros(days)
        day = 0
        done = False
        while not done:
            action = strategy( observation, self._env ) # call strategy
            observation, reward, done, info = self._env.step(action)
            actcnt[action] += 1
            actions[day] = action
            costs[day] = float(info['costs'])
            rewards[day+1] = reward
            navs[day+1] = float(info['nav'])
            pnls[day+1] = float(info['pnl'])
            day += 1
        df = pd.DataFrame( 
                {'action':actions,'reward':rewards,'costs':costs,'pnl':pnls,'nav':navs},
                columns=['action','reward','costs','pnl','nav'])
        return df, actcnt
