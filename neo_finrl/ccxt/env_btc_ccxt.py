import gym
import numpy as np
import numpy.random as rd
import pandas as pd
from gym import spaces
import torch
from elegantrl.agent import AgentDQN


class BitcoinEnv:  # custom env
    def __init__(self, processed_ary, initial_account=10_000, max_stock=1e2, \
                 transaction_fee_percent=0.00075, if_train=True, gamma = 0.99):
        self.stock_dim = 1
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = 1
        self.gamma = gamma
        data_ary = processed_ary

        N = data_ary.shape[0]
        x = int(N*0.995)# ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)

        # assert data_ary.shape == (8640, 13)  
        self.ary_train = data_ary[:x]
        self.ary_valid = data_ary[x:]
        self.ary = self.ary_train if if_train else self.ary_valid


        # reset
        self.day = 0
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.day_npy = self.ary[self.day]
        self.stocks = 0.0  # multi-stack

        self.total_asset = self.account + self.day_npy[3] * self.stocks
        self.episode_return = 0.0  
        self.gamma_return = 0.0
        

        '''env information'''
        self.env_name = 'BitcoinEnv'
        # self.state_dim = 15
        self.state_dim = data_ary.shape[1]+2
        self.action_dim = 3
        self.if_discrete = True
        self.target_return = 1.05 #1.09 # 1.25  # convergence 1.5
        self.max_step = self.ary.shape[0]


    def reset(self) -> np.ndarray:
        self.initial_account__reset = self.initial_account  # reset()
        self.account = self.initial_account__reset
        self.stocks = 0.0
        self.total_asset = 0.0

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((self.account * 2 ** -16, self.day_npy * 2 ** -8, self.stocks * 2 ** -12,)).astype(np.float32)
        return state

    def step(self, action) -> (np.ndarray, float, bool, None):
        if action == 0:
            stock_action = 0
        elif action == 1:
            stock_action = 1
        elif action == 2:
            stock_action = -1
        """bug or sell stock"""
        # print(self.day_npy)
        # close price
        adj = self.day_npy[3]
        if stock_action == 1:  
            if self.stocks <= 0.0:
                available_amount = self.total_asset / adj
                delta_stock = 0.8*available_amount - self.stocks
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks += delta_stock
        elif stock_action == 0: 
            if self.stocks != 0.0:
                delta_stock = self.stocks
                if delta_stock > 0:
                    self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                    self.stocks = 0 
                else:
                    self.account += adj * delta_stock * (1 + self.transaction_fee_percent)
                    self.stocks = 0 
        else:
            if self.stocks >= 0.0:
                available_amount = self.total_asset / adj
                delta_stock = 0.8*available_amount + self.stocks
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks -= delta_stock
            
            

        """update day"""
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  
        state = np.hstack((self.account * 2 ** -16, self.day_npy * 2 ** -8, self.stocks * 2 ** -12,)).astype(np.float32)

        next_total_asset = self.account + self.day_npy[3]*self.stocks
        reward = (next_total_asset - self.total_asset) * 2 ** -16  
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * self.gamma + reward 
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0  
            self.episode_return = next_total_asset / self.initial_account  
        return state, reward, done, None
    
    
    @staticmethod
    def draw_cumulative_return(self, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = AgentDQN()
        net_dim = 2 ** 9
        cwd = './AgentDQN/BitcoinEnv_0'

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()
        btc_returns = list()# the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.max_step):
                if i == 0:
                    init_price = float(state[4])
                s_tensor = _torch.as_tensor((state,), device=device)
                action = act(s_tensor)[0]  # not need detach(), because with torch.no_grad() outside
                a_int = action.argmax(dim=0).cpu().numpy()
                action = a_int 
                state, reward, done, _ = self.step(action)
                
                total_asset = self.account + (self.day_npy[3] * self.stocks)
                episode_return = total_asset / self.initial_account
                episode_returns.append(episode_return)
                btc_return = (state[4]/init_price)
                btc_returns.append(btc_return)
                if done:
                    break

        import matplotlib.pyplot as plt
        plt.plot(episode_returns)
        plt.plot(btc_returns, color = 'yellow')
        plt.grid()
        plt.title('cumulative return')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.show()
        plt.savefig(f'{cwd}/cumulative_return.jpg')
        return episode_returns
