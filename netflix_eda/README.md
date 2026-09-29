# 🎬 Netflix Catalogue Analysis

Exploratory analysis of 8,807 Netflix movies and TV shows (snapshot up to September 2021), with an interactive Streamlit dashboard.

## Questions answered

1. How fast did the catalogue grow, and what's the mix of movies and shows?
2. Who is the content for, and which genres dominate?
3. Which countries produce the content?
4. How long are movies, and how many seasons do shows run?
5. How new is content when it arrives on Netflix?

## Key findings

- **70% movies.** Additions peaked in **2019** at about 2,000 titles.
- **Adult content is the largest audience group** (47% of movies, 43% of shows). Kids' content is 17% of shows but only 7% of movies.
- **The US leads, India is second**, and Indian titles are almost all movies (962 movies vs 84 shows).
- Typical movie: **98 minutes**. **Two thirds** of TV shows have a single season.
- Shows usually arrive in their release year, and movies about **2 years** after release.

## Data cleaning

- Re-encoded the one Latin-1 row. Reading the whole file as ISO-8859-1 (as the first version did) garbled 2,500+ names.
- Dropped 14 empty columns, and moved 3 durations that had been typed into the `rating` column back.
- Excluded 2 rows appended in 2024, outside the snapshot.
- The first version ranked "directors by average rating" by averaging **age ratings** (TV-MA = 18) as if they measured quality.
  Ratings are now grouped into audience bands (Kids, Older kids, Teens, Adults).

## Project structure

| File | Purpose |
|---|---|
| `netflix_data.py` | Loading and cleaning, audience bands, list-column helpers |
| `netflix_eda.ipynb` | Full analysis with conclusions |
| `app.py` | Streamlit dashboard filtered by type and year added |

## Run it

```bash
pip install -r requirements.txt
streamlit run netflix_eda/app.py
```
