import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load your dataset
df = pd.read_csv('data/netflix_titles.csv',encoding= "ISO-8859-1")

# Set page configuration
st.set_page_config(page_title='Netflix Analysis', layout='wide')

st.title('Netflix Data Analysis')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a Visualization:', 
                               ['Content Type Count', 'Rating Distribution', 'Movies Released Each Year', 
                                'Top Directors by Average Rating', 'Top Genres', 'Content Additions by Year',
                                'Top Actors', 'Duration Distribution', 'Content Types by Year'])

# Function to plot Content Type Count
def plot_content_type_count():
    content_values_counts = df['type'].value_counts()
    fig, ax = plt.subplots(figsize=(16,9))
    content_values_counts.plot(kind='bar', color=['blue', 'red'], ax=ax)
    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    ax.set_title('Content Type Count')
    st.pyplot(fig)

def plot_rating_distribution():
    rating_value_counts = df['rating'].value_counts()
    fig, ax = plt.subplots(figsize=(10,6))
    rating_value_counts.plot(kind='barh', color=['red', 'green', 'blue'], ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Rating')
    ax.set_title('Rating Distribution')
    st.pyplot(fig)

def plot_movies_released_each_year():
    release_year_count = df['release_year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10,6))
    release_year_count.plot(kind='line', ax=ax)
    ax.set_title('Number of Movies Released Each Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.grid(True)
    st.pyplot(fig)

# Function to plot Top Directors by Average Rating
def plot_top_directors():
    rating_map = {
        'TV-Y': 0, 'TV-Y7': 7, 'TV-Y7-FV': 7, 'TV-G': 7, 'TV-PG': 12, 
        'TV-14': 14, 'TV-MA': 18, 'R': 18, 'NC-17': 18, 'NR': None
    }
    df['rating_numeric'] = df['rating'].map(rating_map)
    director_ratings = df.groupby('director')['rating_numeric'].agg(['mean', 'count'])
    director_ratings = director_ratings[director_ratings['count'] >= 10]
    director_ratings = director_ratings.sort_values(by='mean', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10,6))
    director_ratings['mean'].plot(kind='bar', color=['red', 'blue', 'green'], ax=ax)
    ax.set_title('Directors with Highest Average Ratings on Netflix (Minimum 10 Entries)')
    ax.set_xlabel('Director')
    ax.set_ylabel('Average Rating')
    ax.set_xticklabels(director_ratings.index, rotation=45, ha='right')
    st.pyplot(fig)

def plot_top_genres():
    genres = df['listed_in'].apply(lambda x: x.split(', '))
    genre_counts = Counter(genre for sublist in genres for genre in sublist)
    top_genres = dict(genre_counts.most_common(10))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(list(top_genres.keys()), list(top_genres.values()), color='red')
    ax.set_title('Top 10 Most Common Genres on Netflix')
    ax.set_xlabel('Count')
    ax.set_ylabel('Genre')
    ax.invert_yaxis()
    st.pyplot(fig)

def plot_content_additions_by_year():
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y')
    df['year_added'] = df['date_added'].dt.year
    content_additions_by_year = df['year_added'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10,6))
    content_additions_by_year.plot(kind='line', marker='o', color='orange', ax=ax)
    ax.set_title('Trends of Content Additions to Netflix (Yearly)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Content Additions')
    ax.grid(True)
    st.pyplot(fig)

def plot_top_actors():
    actors = df['cast'].dropna().apply(lambda x: x.split(', '))
    actor_counts = Counter(actor for sublist in actors for actor in sublist)
    top_actors = dict(actor_counts.most_common(10))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(list(top_actors.keys()), list(top_actors.values()), color='purple')
    ax.set_title('Top 10 Most Frequent Actors on Netflix')
    ax.set_xlabel('Count')
    ax.set_ylabel('Actor')
    ax.invert_yaxis()
    st.pyplot(fig)

def plot_duration_distribution():
    df['duration'] = df['duration'].fillna('0 min')
    df['duration'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if 'min' in x else 0)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(df['duration'], bins=30, color='green', edgecolor='black')
    ax.set_title('Distribution of Movie Durations')
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def plot_content_types_by_year():
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y')
    df['year_added'] = df['date_added'].dt.year
    content_types_by_year = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10,6))
    content_types_by_year.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Content Types Added to Netflix by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Display selected visualization
if options == 'Content Type Count':
    plot_content_type_count()
elif options == 'Rating Distribution':
    plot_rating_distribution()
elif options == 'Movies Released Each Year':
    plot_movies_released_each_year()
elif options == 'Top Directors by Average Rating':
    plot_top_directors()
elif options == 'Top Genres':
    plot_top_genres()
elif options == 'Content Additions by Year':
    plot_content_additions_by_year()
elif options == 'Top Actors':
    plot_top_actors()
elif options == 'Duration Distribution':
    plot_duration_distribution()
elif options == 'Content Types by Year':
    plot_content_types_by_year()
