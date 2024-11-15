import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast  # For safe evaluation of coordinates

# Load Data with Fallback Mechanism
@st.cache_data  # Updated caching method
def load_data():
    try:
        # Try to load data from SQLite database
        db_path = "data/database.sqlite"
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM Tweets", conn)
        conn.close()
    except Exception as e:
        # Fallback to CSV file if database is not available
        st.warning("Database not found. Falling back to CSV file.")
        csv_path = "data/Tweets.csv"
        df = pd.read_csv(csv_path)

    # Ensure tweet_created is datetime
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    
    # Convert tweet_created to be timezone-naive if it's timezone-aware
    if df['tweet_created'].dt.tz is not None:
        df['tweet_created'] = df['tweet_created'].dt.tz_convert(None)
        
    return df

# Function to safely parse coordinates
def parse_coordinates(coord):
    try:
        return ast.literal_eval(coord)
    except (ValueError, SyntaxError, TypeError):
        return None

# Application Title
st.title("Twitter US Airline Sentiment Dashboard")

# Sidebar
st.sidebar.header("Options")
df = load_data()

# Sidebar Filters
airlines = st.sidebar.multiselect(
    "Select Airlines:",
    options=df['airline'].unique(),
    default=df['airline'].unique()
)

sentiments = st.sidebar.multiselect(
    "Select Sentiments:",
    options=df['airline_sentiment'].unique(),
    default=df['airline_sentiment'].unique()
)

# Date Range Filter
start_date = st.sidebar.date_input("Start Date", df['tweet_created'].min().date())
end_date = st.sidebar.date_input("End Date", df['tweet_created'].max().date())

# Confidence Threshold Slider
confidence_threshold = st.sidebar.slider(
    "Minimum Sentiment Confidence", 0.0, 1.0, 0.5
)

# Filter data based on sidebar selections
filtered_data = df[
    (df['airline'].isin(airlines)) & 
    (df['airline_sentiment'].isin(sentiments)) &
    (df['tweet_created'] >= pd.Timestamp(start_date)) &
    (df['tweet_created'] <= pd.Timestamp(end_date)) &
    (df['airline_sentiment_confidence'] >= confidence_threshold)
]

# Show filtered data in the sidebar
st.sidebar.write(f"Displaying {filtered_data.shape[0]} tweets")
if st.sidebar.checkbox("Show Data", False):
    st.write(filtered_data)

# Interactive Bar Plot
st.subheader("Number of Tweets by Airline and Sentiment")
if not filtered_data.empty:
    bar_fig = px.bar(
        filtered_data,
        x='airline',
        color='airline_sentiment',
        barmode='group',
        title="Tweet Counts by Airline and Sentiment",
        labels={'airline_sentiment': 'Sentiment', 'airline': 'Airline'}
    )
    st.plotly_chart(bar_fig)
else:
    st.write("No data available for the selected filters.")

# Pie Chart
st.subheader("Distribution of Sentiments")
if not filtered_data.empty:
    pie_fig = px.pie(
        filtered_data,
        names='airline_sentiment',
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(pie_fig)
else:
    st.write("No data available for the selected filters.")

# Time Series Analysis
st.subheader("Tweet Trends Over Time")
if not filtered_data.empty:
    time_fig = px.line(
        filtered_data,
        x="tweet_created",
        color="airline_sentiment",
        title="Sentiment Trends Over Time",
        labels={'tweet_created': 'Date', 'airline_sentiment': 'Sentiment'}
    )
    st.plotly_chart(time_fig)
else:
    st.write("No data available for the selected filters.")

# Map (if location data exists)
if 'tweet_coord' in df.columns and df['tweet_coord'].notna().any():
    st.subheader("Tweet Locations")
    location_data = filtered_data.dropna(subset=['tweet_coord'])
    if not location_data.empty:
        # Apply safe coordinate parsing
        location_data['coords'] = location_data['tweet_coord'].apply(parse_coordinates)
        location_data = location_data.dropna(subset=['coords'])
        
        # Separate lat and lon for map plotting
        location_data['lat'] = location_data['coords'].apply(lambda x: x[0] if x else None)
        location_data['lon'] = location_data['coords'].apply(lambda x: x[1] if x else None)
        
        map_fig = px.scatter_geo(
            location_data,
            lat='lat',
            lon='lon',
            color="airline_sentiment",
            title="Tweet Locations on Map"
        )
        st.plotly_chart(map_fig)
    else:
        st.write("No location data available for the selected filters.")
else:
    st.write("No location data available.")

# Exclude Words from Word Cloud
excluded_words = st.sidebar.text_input("Exclude words (comma separated)", "flight,airline")
excluded_words = [word.strip() for word in excluded_words.split(",")]

# Word Cloud
st.subheader("Word Clouds for Sentiments")
sentiment = st.radio("Select Sentiment:", options=df['airline_sentiment'].unique())
sentiment_data = filtered_data[filtered_data['airline_sentiment'] == sentiment]

if not sentiment_data.empty:
    words = ' '.join(sentiment_data['text'].astype(str))
    filtered_words = ' '.join([word for word in words.split() if word.lower() not in excluded_words])
    
    try:
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))  # Explicitly create figure and axis
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  # Hide axes
        st.pyplot(fig)  # Pass the Matplotlib figure directly to Streamlit
    except ValueError as e:
        st.write("Error generating word cloud:", e)
else:
    st.write("No text data available for the selected sentiment.")



# Footer
st.write("Built with Streamlit | Data from Twitter US Airline Sentiment Dataset")
