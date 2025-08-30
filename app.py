import pandas as pd
import streamlit as st

st.title("Airbnb Data Explorer")

@st.cache_data
def load_data():
    reviews = pd.read_csv("reviews.zip", compression="zip")
    calendar = pd.read_csv("calendar.zip", compression="zip")
    listings = pd.read_csv("listings.csv")
    return reviews, calendar, listings

reviewsDF, calendarDF, listingsDF = load_data()

st.subheader("Datasets Loaded Successfully")
st.write("Reviews:", reviewsDF.shape)
st.write("Calendar:", calendarDF.shape)
st.write("Listings:", listingsDF.shape)
