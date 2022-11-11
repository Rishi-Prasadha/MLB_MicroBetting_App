import os
import json
from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
# from noah import starter_code 
# to refer to noah's variables it will be: starter_code.variableName
import pandas as pd

# Define and connect a new Web3 provider
w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER_URI")))

################################################################################
# Contract Helper function:
# 1. Loads the contract once using cache
# 2. Connects to the contract using the contract address and ABI
################################################################################

# Cache the contract on load
@st.cache(allow_output_mutation=True)
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


# Load the contract
contract = load_contract()

################################################################################
# Design
################################################################################

st.markdown("# BTTR")
st.markdown("## *The better betting app*")

################################################################################
# Place the bet
################################################################################

accounts = w3.eth.accounts
account = accounts[0]

address = st.text_input("Enter your Ethereum address here")

st.write("DISPLAY ODDS")
st.write("DISPLAY PITCH NUMBER")
st.write("DISPLAY COUNT")

st.markdown("*FF = Fastball, CU = Curveball, CH = Changeup, SL = Slider*")
pitch_type = st.selectbox("Enter what type of pitch is coming next", ('FF', 'CU', 'CH', 'SL'))
bet_amount = st.text_input("Enter how much you want to bet (in ETH)")
payout = 0.98*((1/odds[pitch_type]) * bet_amount)
# or payout = ((1/odds[pitch_type]) * bet_amount) - (0.02)*((1/odds[pitch_type]) * bet_amount)

# Holder dataframe that keeps all bets logged before sending to smart contract
ff_list = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

cu_list = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

ch_list = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

sl_list = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

################################# MAKE BET LOGIC ########################################
st.markdown("If you have the right address and bet amount please press *Make Bet*")
if st.button("Make Bet"):
    # payout = (odds[pitch_type] * 100) * bet_amount

    if pitch_type == "Fastball":
        ff_list["Address"].append(address)
        ff_list["Bet Amount"].append(bet_amount)
        ff_list["Odds"].append(odds[pitch_type])
        ff_list["Payout"].append(payout)
    elif pitch_type == "Curveball":
        cu_list["Address"].append(address)
        cu_list["Bet Amount"].append(bet_amount)
        cu_list["Odds"].append(odds[pitch_type])
        cu_list["Payout"].append(payout)  
    elif pitch_type == "Changeup":
        ch_list["Address"].append(address)
        ch_list["Bet Amount"].append(bet_amount)
        ch_list["Odds"].append(odds[pitch_type])
        ch_list["Payout"].append(payout)
    else:
        sl_list["Address"].append(address)
        sl_list["Bet Amount"].append(bet_amount)
        sl_list["Odds"].append(odds[pitch_type])
        sl_list["Payout"].append(payout)
    
    # Address, bet amount along with total payout with the odds, growing dataframe 

    # Submit the transaction to the smart contract 
    contract.functions.makeBet(address, payout).transact({'from': account, 'gas': 1000000})


################################# NEXT PITCH LOGIC ########################################
st.markdown("When you're ready to move on to the next pitch please press *Next Pitch*")
pitch_count = 0
if st.button("Next Pitch"):
    # iterates through addresses and pays out the winners 
    next_pitch = list(verlander_df['pitch_type'])
    if next_pitch[pitch_count] == "SL":
        sl_address = list(sl_list['Address'])
        sl_payouts = list(sl_list['Payout'])
        i = 0
        for address in sl_address:
            contract.functions.payout(address, sl_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1
    elif next_pitch[pitch_count] == "FF":
        ff_address = list(ff_list['Address'])
        ff_payouts = list(ff_list['Payout'])
        i = 0
        for address in ff_address:
            contract.functions.payout(address, ff_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1
    elif next_pitch[pitch_count] == "CU":
        cu_address = list(cu_list['Address'])
        cu_payouts = list(sl_list['Payout'])
        i = 0
        for address in cu_address:
            contract.functions.payout(address, cu_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1
    elif next_pitch[pitch_count] == "CH":
        ch_address = list(ch_list['Address'])
        ch_payouts = list(sl_list['Payout'])
        i = 0
        for address in ch_address:
            contract.functions.payout(address, ch_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1

    # Need to delete the data held in dataframes

    pitch_count += 1


    # send a transaction that pays out or keeps funds if they lose 

    # transaction to the smart contract to clear current bets

    # Need to update on-screen components of the streamlit

# if st.button("Submit Transaction"):
#     # Send a transaction to the smart contract to make a bet 





# if st.button("Award Certificate"):
    #contract.functions.awardCertificate(student_account, certificate_details).transact({'from': account, 'gas': 1000000})

# All the math with odds happen in the front end streamlit part
# All the payout or lack thereof math happens in the smart contract

# Instead of creating a data structure in the smart contract, we want to hold all current bets in python
# we separate them into different pitch types and whoever wins, we submit another transaction to the smart contract to pay out
## payout function must take an address, on python side we need to loop through addresses and submit a payout transaction for each address

######### QUESTIONS FOR ERIC 
# How do you initialize a smart contract with a ton of ether?
# If you are not working on remix, how do you get a function to pay an account?
# how do you transfer money to the exterior