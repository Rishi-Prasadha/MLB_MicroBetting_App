# Imports
import os
import json
from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import project2 as p2
import pandas as pd
import streamlit as st

# Import the functions from ethereum.py
from functions import w3, w3_2, generate_account8545, generate_account7545, get_balance, send_transaction


w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3_2 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

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

# Generate the Ethereum account
account = generate_account8545(w3) #sender
smart_contract = generate_account7545(w3_2) #receiver

# The Ethereum Account Address
st.text("\n")
st.text("\n")
st.markdown("## Ethereum Account Address")
st.caption('(Pulled from Ganache)')

# Write the Ethereum account address to the Streamlit page
st.write(account.address)

# Display the Etheremum Account balance
st.text("\n")
st.text("\n")
st.markdown("## Ethereum Account Balance:")

# Call the get_balance function and write the account balance to the screen
ether_balance = get_balance(w3, account.address)
st.write(ether_balance)

# An Ethereum Transaction
st.text("\n")
st.text("\n")
st.markdown("## An Ethereum Transaction")

with st.container():
    st.markdown('#### Select what type of pitch you think is coming next:')
    pitch_type = st.selectbox("Enter what type of pitch you think is coming next", ('Fastball', 'Curveball', 'Changeup', 'Slider'), label_visibility = 'hidden')

    st.subheader('Odds of Selected Pitch:')

st.markdown('---')

with st.container():
    st.markdown('#### To make a bet:')
    # address = st.selectbox(label = 'Select your Ethereum account:', options = accounts[:-1])
    bet_amount = st.number_input("Enter how much you want to bet (in ETH)")

################################################################################
# Place the bet
################################################################################

payout = 0.98*((1/p2.odds[pitch_type]) * float(bet_amount))
# or payout = ((1/odds[pitch_type]) * bet_amount) - (0.02)*((1/odds[pitch_type]) * bet_amount)

################################# MAKE BET LOGIC ########################################
st.markdown("If you have the right address and bet amount please press *Make Bet*")
if st.button("Make Bet"):

    # Submit the transaction to the smart contract 
    transaction_hash = send_transaction(w3, account, smart_contract, bet_amount)

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
