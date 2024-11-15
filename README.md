# ✨ Twitter US Airline Sentiment Dashboard  

An interactive data dashboard built with **Streamlit** to visualize, analyze, and explore sentiment data from tweets about U.S. airlines.

---

## 🚀 Features  

### 1. **Tweet Counts by Airline and Sentiment**  
- 📊 **Bar Chart**: View the distribution of tweets across different airlines, categorized by sentiment (positive, neutral, or negative).

### 2. **Sentiment Distribution**  
- 🥧 **Pie Chart**: Summarize positive, neutral, and negative sentiments in a single visualization.

### 3. **Word Clouds**  
- ☁️ **Dynamic Word Clouds**: Generate word clouds for each sentiment category, showcasing frequently mentioned words in tweets.

### 4. **Geographical Map**  
- 🌍 **Interactive Map**: Pinpoint tweet locations with sentiment-based color coding for quick identification.

### 5. **Time-Series Trends**  
- 📈 **Line Chart**: Analyze tweet trends over time for different airlines and sentiments to uncover patterns.

---

## 📝 Dataset and Setup  

### Dataset  
- **Source**: [Twitter US Airline Sentiment Dataset on Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)  
- The dataset contains information about sentiments expressed in tweets regarding U.S. airlines.

### Setup  
1. Place the dataset (`Tweets.csv` or `database.sqlite`) in the `data/` folder.  
2. If using `database.sqlite`, ensure it contains a `Tweets` table with the following fields:  
   - `airline`  
   - `airline_sentiment`  
   - `text`  
   - `tweet_created`  
   - Optional: `tweet_coord`, `negativereason`, etc.  

---

## 🎓 Learn More  

This project is inspired by the Coursera guided project:  
**[Interactive Data Dashboards with Streamlit and Python](https://www.coursera.org/projects/interactive-data-dashboards-with-streamlit)**  

### What You’ll Learn:  
- Building interactive dashboards with **Streamlit**.  
- Loading datasets efficiently.  
- Creating engaging visualizations.  
- Adding interactivity for enhanced user experience.  

---

## 🤝 Contributions  

Contributions are welcome! 🙌 Here’s how you can help:  

1. **Submit a Pull Request**:  
   - Share new features or bug fixes.  

2. **Open an Issue**:  
   - Suggest ideas or report problems.  

Let’s work together to make this project even better!  

---

## 📜 License  

This project is licensed under the **MIT License**, allowing you to:  
- Freely use, modify, and distribute the project with attribution.  

---

