from elegantrl.run import *
from neo_finrl.alpaca.preprocess_alpaca import preprocess
from neo_finrl.alpaca.data_fetch_alpaca import data_fetch

from neo_finrl.alpaca.env_stock_alpaca import StockTradingEnv
from elegantrl.agent import *
import pickle

'''data_fetch'''
#please fill in your own account info
API_KEY = None
API_SECRET = None
APCA_API_BASE_URL = None
stock_list = [
    "AAPL","MSFT","JPM"
]
start = '2021-05-01'
end = '2021-05-10'
time_interval = '1Min'

# API_KEY = "PK9ZDUS6Z4IPK2TMMJ1M"
# API_SECRET = "sKTlBoUCyORn4g0Ju1xHE3iJXlvwdoB7awatNbtq"
# APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
# alpaca_df = data_fetch(API_KEY=API_KEY, API_SECRET=API_SECRET, APCA_API_BASE_URL=APCA_API_BASE_URL,stock_list=stock_list, 
#                 start_date=start, end_date=end,time_interval=time_interval)
url = 'C:/Users/yosou/Documents/BooksWithCode/Binance/Deep Learning/Reinforcement-Learning/NeoFinRL-main/Data/alpaca_df.pkl'
# filepath = open(url, "wb")
# pickle.dump(alpaca_df, filepath)
# filepath.close()

filepath = open(url, "rb")
alpaca_df = pickle.load(filepath)

print(alpaca_df, type(alpaca_df))
alpaca_ary = preprocess(alpaca_df, stock_list)
print(alpaca_ary.shape)
args = Arguments(if_on_policy=True, gpu_id=0)
args.agent = AgentSAC()
'''choose environment'''
args.env = StockTradingEnv(ary = alpaca_ary, if_train=True)
args.env_eval = StockTradingEnv(ary = alpaca_ary, if_train=False)
args.cwd = './AgentSAC/Stock_Alpaca_0'
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
args.break_step = int(5e5)

train_and_evaluate(args)

args = Arguments(agent=None, env=None, gpu_id=0)
args.agent = AgentSAC()
args.env = StockTradingEnv(ary = alpaca_ary, if_train=False)
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
args.if_remove = False
args.cwd = './AgentSAC/Stock_Alpaca_0'
args.init_before_training()
# Draw the graph
StockTradingEnv(ary = alpaca_ary, if_train=False)\
.draw_cumulative_return(args, torch)
