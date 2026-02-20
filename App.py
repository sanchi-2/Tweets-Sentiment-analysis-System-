import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob


# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)


# ----------------------------
# Title
# ----------------------------
st.title("📊 Twitter Sentiment Analysis System")
st.write("Analyze Individual Tweets + Dataset")


# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z ]+", "", text)
    return text.lower()


# ----------------------------
# Sentiment
# ----------------------------
def analyze_sentiment(text):

    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "😊 Positive"
    elif analysis.sentiment.polarity < 0:
        return "😡 Negative"
    else:
        return "😐 Neutral"


# ----------------------------
# SECTION 1: Single Tweet
# ----------------------------
st.subheader("🔍 Analyze a Single Tweet")

user_input = st.text_area(
    "Paste a tweet here:",
    height=100
)

if st.button("Analyze Tweet"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:

        clean = clean_text(user_input)
        result = analyze_sentiment(clean)

        st.success("Sentiment Result:")
        st.markdown(f"### {result}")


st.markdown("---")


# ----------------------------
# SECTION 2: Dataset Analysis
# ----------------------------
st.subheader("📈 Analyze Offline Dataset")


@st.cache_data
def load_data():

    df = pd.read_csv(
        "training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        header=None
    )

    df.columns = [
        "Target", "ID", "Date", "Flag", "User", "Tweet"
    ]

    df = df.sample(5000)

    df["Clean"] = df["Tweet"].apply(clean_text)
    df["Sentiment"] = df["Clean"].apply(analyze_sentiment)

    return df


if st.button("Analyze Dataset"):

    with st.spinner("Processing dataset..."):

        df = load_data()


    # Count
    pos = len(df[df["Sentiment"] == "😊 Positive"])
    neg = len(df[df["Sentiment"] == "😡 Negative"])
    neu = len(df[df["Sentiment"] == "😐 Neutral"])


    # Metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("😊 Positive", pos)
    c2.metric("😡 Negative", neg)
    c3.metric("😐 Neutral", neu)


    # Chart
    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots()

    ax.pie(
        [pos, neg, neu],
        labels=["Positive", "Negative", "Neutral"],
        autopct="%1.1f%%",
        startangle=90
    )

    st.pyplot(fig)


    # Table
    st.subheader("Sample Tweets")

    st.dataframe(
        df[["Tweet", "Sentiment"]].head(20),
        use_container_width=True
    )


# Footer
st.markdown("---")
st.markdown("Developed by sanchi 🚀")
