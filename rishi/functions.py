import os
import json
from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import project2 as p2
from bip44 import Wallet
from web3 import Account
import pandas as pd
from web3.gas_strategies.time_based import medium_gas_price_strategy


################################################################################
# Contract Helper function:
# 1. Loads the contract once using cache
# 2. Connects to the contract using the contract address and ABI
################################################################################

w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3_2 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

# Cache the contract on load
# @st.cache(allow_output_mutation=True)
# Define the load_contract function
def load_contract():

    # Load Art Gallery ABI
    with open(Path('./contracts/compiled/certificate_abi.json')) as f:
        certificate_abi = json.load(f)

    # Set the contract address (this is the address of the deployed contract)
    contract_address = os.getenv("SMART_CONTRACT_ADDRESS")

    # Get the contract
    contract = w3.eth.contract(
        address=contract_address,
        abi=certificate_abi
    )
    # Return the contract from the function
    return contract

def generate_account8545(w3):
    """Create a digital wallet and Ethereum account from a mnemonic seed phrase."""
    # Access the mnemonic phrase from the `.env` file
    mnemonic = os.getenv("MNEMONIC8545")

    # Create Wallet object instance
    wallet = Wallet(mnemonic)

    # Derive Ethereum private key
    private, public = wallet.derive_account("eth")

    # Convert private key into an Ethereum account
    account = Account.privateKeyToAccount(private)

    # Return the account from the function
    return account

def generate_account7545(w3):
    """Create a digital wallet and Ethereum account from a mnemonic seed phrase."""
    # Access the mnemonic phrase from the `.env` file
    mnemonic = os.getenv("MNEMONIC7545")

    # Create Wallet object instance
    wallet = Wallet(mnemonic)

    # Derive Ethereum private key
    private, public = wallet.derive_account("eth")

    # Convert private key into an Ethereum account
    account = Account.privateKeyToAccount(private)

    # Return the account from the function
    return account

def get_balance(w3, address):
    """Using an Ethereum account address access the balance of Ether"""
    # Get balance of address in Wei
    wei_balance = w3.eth.get_balance(address)

    # Convert Wei value to ether
    ether = w3.fromWei(wei_balance, "ether")

    # Return the value in ether
    return ether


# Create a function called `send_transaction` that creates a raw transaction, signs it, and sends it. Return the confirmation hash from the transaction
def send_transaction(w3, account, receiver, ether):
    """Send an authorized transaction."""
    # Set a medium gas price strategy
    w3.eth.setGasPriceStrategy(medium_gas_price_strategy)

    # Convert eth amount to Wei
    wei_value = w3.toWei(ether, "ether")

    # Calculate gas estimate
    gas_estimate = w3.eth.estimateGas({"to": receiver.address, "from": account.address, "value": wei_value})

    # Construct a raw transaction
    raw_tx = {
        "to": receiver.address,
        "from": account.address,
        "value": wei_value,
        "gas": gas_estimate,
        "gasPrice": 0,
        "nonce": w3.eth.getTransactionCount(account)
    }

    # Sign the raw transaction with ethereum account
    signed_tx = account.signTransaction(raw_tx)

    # Send the signed transactions
    return w3.eth.sendRawTransaction(signed_tx.rawTransaction)