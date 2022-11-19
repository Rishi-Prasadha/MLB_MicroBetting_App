# MLB MicroBetting App

The worldwide sports betting market is at a whopping $3 trillion. Behind the NFL and college football, the NBA, MLB and NHL make up the second largest market for sports betting at $80 Billion. Although there are many companies in the cryptocurrency sports betting arena, the area of micro-betting--the act of betting on real-time small, repetitive but key actions of players in the game--has not been explored yet. We are looking to create an application that will allow bettors to bet on Justin Verlander's pitches, pitch by pitch, aided by our in-house, hyper-optimized machine learning models. 

---

## Technologies 

This application was written on Python. For the backend, blockchain application, the program used the Web3 and BIP libraries. Ganache was used as a local test network that had Ethereum wallets we can interact with. For the frontend, we used Streamlit. Please refer to the following:

* [Ganache](https://trufflesuite.com/ganache/)
* [Web3](https://web3py.readthedocs.io/en/v5/)
* [BIP](https://pypi.org/project/bip-utils/)
* [Streamlit](https://streamlit.io)


For the hyper-optimization part of the program, we used the following libraries: 
* [Pandas](https://github.com/pandas-dev/pandas)
* [Numpy](https://github.com/numpy/numpy)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [TensorFlow](https://www.tensorflow.org/api_docs)

---

## Installation Guide

Before running the above libraries, you will need to install them first. Please refer to the following install prompts in terminal as needed:


`pip install pandas`

`pip install numpy`

`pip install tensorflow`

`pip install -U scikit-learn`

`npm install web3`

`pip install streamlit`

`pip install bip_utils`

`pip install scikit-optimize`

In order to install Ganache, please follow the hyperlink under the *Technologies* section.

---

## Libraries 

This application include many aspects of Web3 and BIP libraries. Please refer to the imports below for the functions we wrote for this application:

```python
# Imports
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from bip44 import Wallet
from web3 import Account
from web3 import middleware
from web3.gas_strategies.time_based import medium_gas_price_strategy
from web3 import Web3
```

Additionally, for the Streamlit application itself, refer to the imports below: 

```python 
# Imports
import os
import json
from web3 import Web3, Account
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import project2 as p2
import pandas as pd

# Import the functions from ethereum.py
from ethereum import generate_account_sender, get_balance, send_transaction
```

```python
# Imports
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
```
---

## Database

All the data used was sourced and downloaded from [Baseball Savant](https://baseballsavant.mlb.com). Thank you to Baseball Savant for providing reliable, accurate player, team, and game statistics after each game. 

With that being said, the Justin Verlander dataset used is found in the resources folder, including the September 29th game data: 

* verlander_update.csv
* field_test_data.csv

---

## Usage 

Our application utilizes the Streamlit library as a frontend where users and bettors can interact with. In order to use the final versions of our application, please refer to the *testing* folder, and run the follwoing command:

`streamlit run streamlit.py`

Please ensure that our application is connected with your Ganache server with the appropriate RPC server and account addresses. 

---

## Results

### Alessandro's Neural Network:
After running both tests on new data, they have never seen we can see that the hyper-optimized neural network model increased its accuracy score by 4%. It may not be such a high increase, but it shows that the hyper-optimized neural network could predict better than the default model. However, this model still needs a lot of optimization and work, especially since it isn't consistent with the outputs. This issue is likely caused to not having powerful software such as AWS or not finding the best parameters for the model. We can also notice that the precision section of both results seems skewed, which may be due to insufficient data. With more time on researching documentation for hyper-optimizing neural network on a software system rather than locally, we could produce a model that may have an accuracy score higher than the hyper-optimized model used here.


--- 

## Contributors

[Brittanie Polasek](https://www.linkedin.com/in/brittanie-polasek/), [Rishi Prasadha](https://www.linkedin.com/in/rishi-prasadha-912212133/), Noah Saleh, Jake Wheeler, [Alessandro Valentini](https://www.linkedin.com/in/alex-valentini-29539a1a9/)

---

## License

MIT
