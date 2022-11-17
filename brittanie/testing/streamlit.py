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

w3_better = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

#Set streamlit page customization
st.set_page_config(page_title = 'BTTR', page_icon = './images/dice.png')

# Streamlit application headings
with st.container():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image('./images/dice.png')

    with col2:
        st.title("BTTR")
        
    st.caption("## *The better betting app*")

st.markdown('---')

#Display latest pitch number and pitch count
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader('Current Pitch Number:')
    st.subheader(p2.verlander_df['pitch_count'][0])
with col2:
    st.subheader('Current Pitch Count:')
    st.subheader(p2.verlander_df['count'][0])

st.markdown('---')

#Display odds and have better choose what type of pitch is coming next
with st.container():
    st.markdown('#### Select what type of pitch you think is coming next:')
    pitch_type = st.selectbox("Enter what type of pitch you think is coming next", ('Fastball', 'Curveball', 'Changeup', 'Slider'), label_visibility = 'hidden')

    st.subheader('Odds of Selected Pitch:')
    st.dataframe(p2.odds.iloc[0])
st.markdown('---')

# Generate the Ethereum account
better_account = generate_account_sender(w3_better)

# The Ethereum Account Address
st.markdown('## To make a bet:')
st.markdown("### Your Ethereum Account Address")
st.caption('(Pulled from Ganache)')

# Write the Ethereum account address to the Streamlit page
st.write(better_account.address)

# Display the Etheremum Account balance
st.markdown("## Your Ethereum Account Balance:")

# Call the get_balance function and write the account balance to the screen
ether_balance = get_balance(w3_better, better_account.address)
st.write(ether_balance)

# An Ethereum Transaction
st.markdown("## Make a Bet")

# Create inputs for the receiver address and ether amount
company_account = Account.privateKeyToAccount(os.getenv("company_private_key"))
ether = st.number_input("Input the amount of ether you want to bet.")

# Create a button that calls the `send_transaction` function and returns the transaction hash
if st.button("Make Bet"):

    transaction_hash = send_transaction(w3_better, better_account, company_account, ether)

    # Display the Etheremum Transaction Hash
    st.markdown("## Ethereum Transaction Hash:")

    st.write(transaction_hash)

payout = 0.98*((1/p2.odds[pitch_type]) * float(ether))

# payout = 0.98*(float(ether))

if st.button("Check Results"):
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
    if next_pitch[0] == pitch_type:
        transaction_hash = send_transaction(w3_better, company_account, better_account, payout)
        st.write(payout)

        # Display the Etheremum Transaction Hash
        st.text("\n")
        st.text("\n")
        st.markdown("# Congratulations, you won!")
        st.balloons()
        st.markdown("## Ethereum Transaction Hash:")
        st.write(transaction_hash)
    else:
        st.markdown("## You lost. Better luck next time")

    pitch_count += 1
