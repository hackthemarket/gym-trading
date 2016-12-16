import gym
from gym import error, spaces, utils
from gym.utils import seeding

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
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.src = QuandlEnvSrc() 
    self.trading_cost_bps = 1e-3
    self.time_cost_bps    = 1e-4
    self.action_space = spaces.Discrete( 3 )
    #pdb.set_trace()
    self.observation_space= spaces.Box( self.src.min_values,
                                        self.src.max_values)#, shape=(5,) )
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

    #if abs(daypnl) > .1:
    #pdb.set_trace()
      
    if self.nav <= LoserNAV:
      done = True
      reward = -1
      #log.debug('loss')
    if self.nav >= WinnerNAV:
      done = True
      reward = 1
      #log.debug('win!')

    #if done:
      #log.info('final NAV: %f', self.nav)
      
    info = { 'daypnl': daypnl, 'nav':self.nav, 'costs':costs }

    return observation, reward, done, info
  
  def _reset(self):
    self.position = 0.0
    self.nav = 1.0
    self.src.reset()
    return self.src.step()[0]
    
  def _render(self, mode='human', close=False):
    #... TODO
    pass
  
