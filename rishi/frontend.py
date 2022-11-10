import os
import json
from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
# from noah import starter_code 
# to reer to noah's variables it will be: starter_code.variableName

# Define and connect a new Web3 provider
w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER_URI")))

################################################################################
# Contract Helper function:
# 1. Loads the contract once using cache
# 2. Connects to the contract using the contract address and ABI
################################################################################

# # Cache the contract on load
# @st.cache(allow_output_mutation=True)
# # Define the load_contract function
# def load_contract():

#     # Load Art Gallery ABI
#     with open(Path('./contracts/compiled/certificate_abi.json')) as f:
#         certificate_abi = json.load(f)

#     # Set the contract address (this is the address of the deployed contract)
#     contract_address = os.getenv("SMART_CONTRACT_ADDRESS")

#     # Get the contract
#     contract = w3.eth.contract(
#         address=contract_address,
#         abi=certificate_abi
#     )
#     # Return the contract from the function
#     return contract


# # Load the contract
# contract = load_contract()

################################################################################
# Design
################################################################################

st.markdown("# BTTR")
st.markdown("## *The better betting app*")

################################################################################
# Place the bet
################################################################################

address = st.text_input("Enter your Ethereum address here")

st.write("DISPLAY ODDS")
st.write("DISPLAY PITCH NUMBER")
st.write("DISPLAY COUNT")

pitch_type = st.selectbox("Enter what type of pitch is coming next", ('Fastball', 'Curveball', 'Changeup', 'Slider'))
bet_amount = st.text_input("Enter how much you want to bet (in ETH)")


if st.button("Next Pitch"):
    # send a transaction that pays out or keeps funds if they lose 

    # transaction to the smart contract to clear current bets

    # Need to update on-screen components of the streamlit

if st.button("Submit Transaction"):
    # Send a transaction to the smart contract to make a bet 


# if st.button("Award Certificate"):
#     contract.functions.awardCertificate(student_account, certificate_details).transact({'from': account, 'gas': 1000000})

# All the math with odds happen in the front end streamlit part
# All the payout or lack thereof math happens in the smart contract