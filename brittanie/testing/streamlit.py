# Imports
import streamlit as st

# Import the functions from ethereum.py
from ethereum import w3, w3_2, generate_account8545, generate_account7545, get_balance, send_transaction
from web3 import Web3

w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3_2 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

# Streamlit application headings
st.markdown("# Automating Ethereum with Streamlit!")

# Generate the Ethereum account
account = generate_account8545(w3) #sender
account2 = generate_account7545(w3_2) #receiver

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

# Create inputs for the receiver address and ether amount
ether = st.number_input("Input the amount of ether")

# Create a button that calls the `send_transaction` function and returns the transaction hash
if st.button("Send Transaction"):

    transaction_hash = send_transaction(w3, account, account2, ether)

    # Display the Etheremum Transaction Hash
    st.text("\n")
    st.text("\n")
    st.markdown("## Ethereum Transaction Hash:")

    st.write(transaction_hash)