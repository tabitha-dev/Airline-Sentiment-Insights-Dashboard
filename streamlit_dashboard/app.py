import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
import re  # For safer word exclusion parsing

# Load Data with Fallback Mechanism
@st.cache_data
def load_data():
    try:
        db_path = "data/database.sqlite"
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM Tweets", conn)
        conn.close()
    except Exception:
        st.warning("Database not found. Falling back to CSV file.")
        csv_path = "data/Tweets.csv"
        df = pd.read_csv(csv_path)

    # Ensure tweet_created is datetime
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    if df['tweet_created'].dt.tz is not None:
        df['tweet_created'] = df['tweet_created'].dt.tz_convert(None)

    return df

# Function to safely parse coordinates
def parse_coordinates(coord):
    try:
        return ast.literal_eval(coord)
    except (ValueError, SyntaxError, TypeError):
        return None

# Title and styling
st.markdown(
    "<style>h1 {text-align: center; color: navy;}</style>",
    unsafe_allow_html=True
)
st.title("Twitter US Airline Sentiment Dashboard")

# Sidebar
st.sidebar.header("Options")
df = load_data()

# Sidebar Filters
airlines = st.sidebar.multiselect(
    "Select Airlines:",
    options=df['airline'].unique(),
    default=df['airline'].unique(),
    help="Filter tweets by airline"
)

sentiments = st.sidebar.multiselect(
    "Select Sentiments:",
    options=df['airline_sentiment'].unique(),
    default=df['airline_sentiment'].unique(),
    help="Filter tweets by sentiment"
)

start_date = st.sidebar.date_input("Start Date", df['tweet_created'].min().date())
end_date = st.sidebar.date_input("End Date", df['tweet_created'].max().date())

confidence_threshold = st.sidebar.slider(
    "Minimum Sentiment Confidence", 0.0, 1.0, 0.5
)

# Filter data
filtered_data = df.query(
    "airline in @airlines and "
    "airline_sentiment in @sentiments and "
    "tweet_created >= @start_date and "
    "tweet_created <= @end_date and "
    "airline_sentiment_confidence >= @confidence_threshold"
)

st.sidebar.write(f"Displaying {filtered_data.shape[0]} tweets")
if st.sidebar.checkbox("Show Data", False):
    st.subheader("Filtered Data")
    st.write(filtered_data)

# Overview Section
st.subheader("Number of Tweets by Airline and Sentiment")
if not filtered_data.empty:
    bar_fig = px.bar(
        filtered_data,
        x='airline',
        color='airline_sentiment',
        barmode='group',
        title="Tweet Counts by Airline and Sentiment",
        labels={'airline_sentiment': 'Sentiment', 'airline': 'Airline'},
        color_discrete_sequence=px.colors.qualitative.Set2  # Vibrant color palette
    )
    st.plotly_chart(bar_fig)
else:
    st.write("No data available for the selected filters.")

st.subheader("Distribution of Sentiments")
if not filtered_data.empty:
    pie_fig = px.pie(
        filtered_data,
        names='airline_sentiment',
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(pie_fig)
else:
    st.write("No data available for the selected filters.")

# Trends Section
st.subheader("Sentiment Confidence Trends Over Time")
if not filtered_data.empty:
    time_fig = px.line(
        filtered_data,
        x="tweet_created",
        y="airline_sentiment_confidence",
        color="airline_sentiment",
        title="Sentiment Confidence Trends Over Time",
        labels={'tweet_created': 'Date', 'airline_sentiment_confidence': 'Confidence'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(time_fig)
else:
    st.write("No data available for the selected filters.")

# Word Cloud Section
excluded_words = set(re.split(r"\s*,\s*", st.sidebar.text_input("Exclude words (comma separated)", "flight,airline")))
st.subheader("Word Cloud by Sentiment")
sentiment = st.radio("Select Sentiment:", options=df['airline_sentiment'].unique())
sentiment_data = filtered_data[filtered_data['airline_sentiment'] == sentiment]

if not sentiment_data.empty:
    words = ' '.join(sentiment_data['text'].astype(str))
    filtered_words = ' '.join([word for word in words.split() if word.lower() not in excluded_words])

    try:
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="black",
            colormap="Set3"
        ).generate(filtered_words)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except ValueError as e:
        st.write("Error generating word cloud:", e)
else:
    st.write("No text data available for the selected sentiment.")

# Map Section
st.subheader("Tweet Locations")
if 'tweet_coord' in df.columns and df['tweet_coord'].notna().any():
    location_data = filtered_data.dropna(subset=['tweet_coord'])
    if not location_data.empty:
        location_data['coords'] = location_data['tweet_coord'].apply(parse_coordinates)
        location_data = location_data.dropna(subset=['coords'])
        location_data['lat'] = location_data['coords'].apply(lambda x: x[0] if x else None)
        location_data['lon'] = location_data['coords'].apply(lambda x: x[1] if x else None)

        map_fig = px.scatter_geo(
            location_data,
            lat='lat',
            lon='lon',
            color="airline_sentiment",
            title="Detailed Tweet Locations Map",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            projection="orthographic"
        )

        map_fig.update_layout(
            geo=dict(
                showland=True,
                landcolor="white",
                showocean=True,
                oceancolor="lightblue",
                showcountries=True,
                countrycolor="gray",
                showsubunits=True,
                subunitcolor="gray",
                showcoastlines=True,
                coastlinecolor="darkgray",
            ),
            title_font_size=20,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        st.plotly_chart(map_fig)
    else:
        st.write("No location data available for the selected filters.")
else:
    st.write("No location data available.")

# Footer
st.write("Built with Streamlit | Data from Twitter US Airline Sentiment Dataset")
