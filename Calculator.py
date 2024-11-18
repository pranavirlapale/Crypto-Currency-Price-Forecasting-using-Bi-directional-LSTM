import streamlit as st
import requests


def app():
    
    st.title("Cryptocurrency Calculator")

   
    crypto_name = st.text_input("Enter a cryptocurrency name (e.g., Bitcoin, Ethereum):")
    crypto_name = crypto_name.strip()
    crypto_name = crypto_name.lower()

    if crypto_name:
        st.write(f"You've selected {crypto_name.capitalize()}")
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_name}&vs_currencies=usd"
        response = requests.get(url)
        data = response.json()

        if crypto_name in data:
            price_in_usd = data[crypto_name]['usd']
            st.write(f"Price in USD: {price_in_usd}")

            amount = st.number_input("Enter the amount of cryptocurrency:", min_value=0.0)

            if amount:
                converted_amount = amount * price_in_usd
                st.write(f"{amount} {crypto_name.capitalize()} is equal to {converted_amount:.2f} USD")
        else:
            st.write("Cryptocurrency not found")
        

