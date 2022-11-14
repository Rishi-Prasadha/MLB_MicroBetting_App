import os
import json
from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import project2 as p2
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
st.set_page_config(page_title = 'BTTR', page_icon = '../images/dice.png')

with st.container():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image('../images/dice.png')

    with col2:
        st.title("BTTR")
        
    st.caption("## *The better betting app*")

st.markdown('---')

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader('Current Pitch Number:')
with col2:
    st.subheader('Current Pitch Count:')

st.markdown('---')

with st.container():
    st.markdown('#### Select what type of pitch you think is coming next:')
    pitch_type = st.selectbox("Enter what type of pitch you think is coming next", ('Fastball', 'Curveball', 'Changeup', 'Slider'), label_visibility = 'hidden')

    st.subheader('Odds of Selected Pitch:')

st.markdown('---')

with st.container():
    st.markdown('#### To make a bet:')
    address = st.text_input("Enter your Ethereum address here")
    bet_amount = st.text_input("Enter how much you want to bet (in ETH)")

################################################################################
# Place the bet
################################################################################

accounts = w3.eth.accounts
account = accounts[0]

payout = 0.98*((1/p2.odds[pitch_type]) * bet_amount)
# or payout = ((1/odds[pitch_type]) * bet_amount) - (0.02)*((1/odds[pitch_type]) * bet_amount)

# Holder dataframe that keeps all bets logged before sending to smart contract
ff_df = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

cu_df = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

ch_df = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

sl_df = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
})

################################# MAKE BET LOGIC ########################################
st.markdown("If you have the right address and bet amount please press *Make Bet*")
if st.button("Make Bet"):

    # Store bets in dataframes
    if pitch_type == "Fastball":
        ff_df["Address"].append(address)
        ff_df["Bet Amount"].append(bet_amount)
        ff_df["Odds"].append(odds[pitch_type])
        ff_df["Payout"].append(payout)
    elif pitch_type == "Curveball":
        cu_df["Address"].append(address)
        cu_df["Bet Amount"].append(bet_amount)
        cu_df["Odds"].append(odds[pitch_type])
        cu_df["Payout"].append(payout)  
    elif pitch_type == "Changeup":
        ch_df["Address"].append(address)
        ch_df["Bet Amount"].append(bet_amount)
        ch_df["Odds"].append(odds[pitch_type])
        ch_df["Payout"].append(payout)
    else:
        sl_df["Address"].append(address)
        sl_df["Bet Amount"].append(bet_amount)
        sl_df["Odds"].append(odds[pitch_type])
        sl_df["Payout"].append(payout)

    # Submit the transaction to the smart contract 
    contract.functions.makeBet(address, payout, bet_amount).transact({'from': account, 'gas': 1000000})


################################# NEXT PITCH LOGIC ########################################
st.markdown("When you're ready to move on to the next pitch please press *Next Pitch*")
pitch_count = 0
if st.button("Next Pitch"):
    # iterates through addresses and pays out the winners 
    next_pitch = list(verlander_df['pitch_type'])
    if next_pitch[pitch_count] == "SL":
        sl_address = list(sl_df['Address'])
        sl_payouts = list(sl_df['Payout'])
        i = 0
        for address in sl_address:
            contract.functions.payout(address, sl_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1
    elif next_pitch[pitch_count] == "FF":
        ff_address = list(ff_df['Address'])
        ff_payouts = list(ff_df['Payout'])
        i = 0
        for address in ff_address:
            contract.functions.payout(address, ff_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1
    elif next_pitch[pitch_count] == "CU":
        cu_address = list(cu_df['Address'])
        cu_payouts = list(cu_df['Payout'])
        i = 0
        for address in cu_address:
            contract.functions.payout(address, cu_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1
    elif next_pitch[pitch_count] == "CH":
        ch_address = list(ch_df['Address'])
        ch_payouts = list(ch_df['Payout'])
        i = 0
        for address in ch_address:
            contract.functions.payout(address, ch_payouts[i]).transact({'from':account, 'gas': 1000000})
            i += 1

    # Need to delete and reset the data held in dataframes
    ff_df = pd.DataFrame({
    "Address": [],
    "Bet Amount": [],
    "Odds": [],
    "Payout": []
    })

    cu_df = pd.DataFrame({
        "Address": [],
        "Bet Amount": [],
        "Odds": [],
        "Payout": []
    })

    ch_df = pd.DataFrame({
        "Address": [],
        "Bet Amount": [],
        "Odds": [],
        "Payout": []
    })

    sl_df = pd.DataFrame({
        "Address": [],
        "Bet Amount": [],
        "Odds": [],
        "Payout": []
    })

    pitch_count += 1

    # send a transaction that pays out or keeps funds if they lose 

#     # transaction to the smart contract to clear current bets

#     # Need to update on-screen components of the streamlit

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