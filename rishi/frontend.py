import os
import json
from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import project2 as p2
# to refer to noah's variables it will be: starter_code.variableName
import pandas as pd
from functions import load_contract, send_transaction

# Define and connect a new Web3 provider
w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER_URI")))

# Load the contract
contract = load_contract()

# Create accounts from Ganache
accounts = w3.eth.accounts
account = accounts[0]
smart_contract = accounts[-1]

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
    st.subheader(p2.verlander_df['pitch_count'][0])
with col2:
    st.subheader('Current Pitch Count:')
    st.subheader(p2.verlander_df['count'][0])

st.markdown('---')

with st.container():
    st.markdown('#### Select what type of pitch you think is coming next:')
    pitch_type = st.selectbox("Enter what type of pitch you think is coming next", ('Fastball', 'Curveball', 'Changeup', 'Slider'), label_visibility = 'hidden')

    st.subheader('Odds of Selected Pitch:')

st.markdown('---')

with st.container():
    st.markdown('#### To make a bet:')
    address = st.selectbox(label = 'Select your Ethereum account:', options = accounts[:-1])
    bet_amount = int(st.number_input("Enter how much you want to bet (in ETH)"))

################################################################################
# Place the bet
################################################################################

payout = 0.98*((1/p2.odds[pitch_type]) * float(bet_amount))
# or payout = ((1/odds[pitch_type]) * bet_amount) - (0.02)*((1/odds[pitch_type]) * bet_amount)

################################# MAKE BET LOGIC ########################################
st.markdown("If you have the right address and bet amount please press *Make Bet*")
if st.button("Make Bet"):

    # Submit the transaction to the smart contract 
    transaction_hash = send_transaction(w3, address, smart_contract, bet_amount)

    # Display the Etheremum Transaction Hash
    st.text("\n")
    st.text("\n")
    st.markdown("## Ethereum Transaction Hash:")

    st.write(transaction_hash)



################################# NEXT PITCH LOGIC ########################################
st.markdown("When you're ready to move on to the next pitch please press *Next Pitch*")

if st.button("Next Pitch"):
    pitch_count = 0

    if pitch_type == "Fastball":
        pitch_type = "FF"
    elif pitch_type == "Curveball":
        pitch_type = "CU"
    elif pitch_type == "Changeup":
        pitch_type = "CH"
    else: 
        pitch_type = "SL"
    
    next_pitch = list(p2.field_df['pitch_type'])
    if next_pitch(pitch_count) == pitch_type:
        transaction_hash = send_transaction(w3, smart_contract, address, payout)

        # Display the Etheremum Transaction Hash
        st.text("\n")
        st.text("\n")
        st.markdown("Congratulations, you won!")
        st.markdown("## Ethereum Transaction Hash:")
        st.write(transaction_hash)
    else:
        st.markdown("## You suck, you lost. Better luck next time")

    pitch_count += 1


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