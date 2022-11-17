# MLB_MicroBetting_App

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

---

## Database

---

## Usage 

---

## Results

--- 

## Contributors

[Brittanie Polasek](https://www.linkedin.com/in/brittanie-polasek/), [Rishi Prasadha](https://www.linkedin.com/in/rishi-prasadha-912212133/), Noah Saleh, Jake Wheeler, Alessandro Valentini

---

## License

MIT