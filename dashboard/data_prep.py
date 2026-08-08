import pandas as pd
import plotly.graph_objects as go

def load_dashboard_data(interim_path: str = "data/interim"):
    """Loads the data the dashboard needs. Uses ratings_val (not
    ratings_val_warm) since dashboard users don't need to be
    'warm' by the meta-model's definition -- any user with enough
    validation-split history to show a meaningful profile works."""
    movies_master = pd.read_parquet(f"{interim_path}/movies_master.parquet")
    ratings_val = pd.read_parquet(f"{interim_path}/ratings_val.parquet")
    return movies_master, ratings_val


def get_user_genre_profile(user_id: int, ratings_val: pd.DataFrame, movies_master: pd.DataFrame,
                            min_rating: float = 3.5) -> pd.Series:
    """
    Builds a genre preference profile for one user: counts how many
    movies they rated >= min_rating per genre, from VALIDATION-split
    ratings only (spec Section 6 -- never test).
    """
    user_ratings = ratings_val[(ratings_val["userId"] == user_id) & (ratings_val["rating"] >= min_rating)]
    user_movies = movies_master[movies_master["movieId"].isin(user_ratings["movieId"])]

    genre_counts = {}
    for genre_list in user_movies["genre_list"]:
        for genre in genre_list:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        # Fall back to MovieLens genres if TMDB genre_list is empty (per Week 1's known gap)
    for _, row in user_movies[user_movies["genre_list"].str.len() == 0].iterrows():
        for genre in row.get("ml_genre_list", []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    return pd.Series(genre_counts).sort_values(ascending=False)

GENRE_NAME_MAP = {
    "Sci-Fi": "Science Fiction",
    "Film-Noir": "Film Noir",
    "IMAX": None,  # not a genre, drop it if it ever appears
}

def normalize_genre_name(genre: str) -> str:
    mapped = GENRE_NAME_MAP.get(genre, genre)
    return mapped


def get_user_genre_profile(user_id: int, ratings_val: pd.DataFrame, movies_master: pd.DataFrame,
                            min_rating: float = 3.5) -> pd.Series:
    user_ratings = ratings_val[(ratings_val["userId"] == user_id) & (ratings_val["rating"] >= min_rating)]
    user_movies = movies_master[movies_master["movieId"].isin(user_ratings["movieId"])]

    genre_counts = {}

    def add_genres(genre_list):
        for genre in genre_list:
            norm = normalize_genre_name(genre)
            if norm is not None:
                genre_counts[norm] = genre_counts.get(norm, 0) + 1

    for genre_list in user_movies["genre_list"]:
        add_genres(genre_list)
    for _, row in user_movies[user_movies["genre_list"].str.len() == 0].iterrows():
        add_genres(row.get("ml_genre_list", []))

    return pd.Series(genre_counts).sort_values(ascending=False)

def build_genre_radar_chart(genre_profile: pd.Series, top_n: int = 8, title: str = "Genre Preferences"):
    
    top_genres = genre_profile.head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=top_genres.values,
        theta=top_genres.index,
        fill='toself',
        name='Genre Preference',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, top_genres.max() * 1.1])),
        showlegend=False,
        title=title,
    )
    return fig

def get_user_decade_profile(user_id: int, ratings_val: pd.DataFrame, movies_master: pd.DataFrame,
                             min_rating: float = 3.5) -> pd.Series:
    """
    Counts how many movies a user rated highly per decade, from
    VALIDATION-split ratings only. Uses movies_master's `year` column
    (parsed from MovieLens titles back in Week 1).
    """
    user_ratings = ratings_val[(ratings_val["userId"] == user_id) & (ratings_val["rating"] >= min_rating)]
    user_movies = movies_master[movies_master["movieId"].isin(user_ratings["movieId"])].copy()

    # Drop movies with unknown year (small edge case from Week 1 -- 412 movies, 0.66%)
    user_movies = user_movies.dropna(subset=["year"])
    user_movies["decade"] = (user_movies["year"] // 10 * 10).astype(int)

    decade_counts = user_movies["decade"].value_counts().sort_index()
    return decade_counts


def build_decade_bar_chart(decade_profile: pd.Series, title: str = "Decade Preferences"):
    """Bar chart, not radar -- decades have a natural chronological order
    that a bar chart preserves visually, unlike a radar chart's circular
    layout which would obscure the timeline."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{d}s" for d in decade_profile.index],
        y=decade_profile.values,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Decade",
        yaxis_title="Movies Rated Highly",
    )
    return fig

def get_user_cast_affinity(user_id: int, ratings_val: pd.DataFrame, movies_master: pd.DataFrame,
                            min_rating: float = 3.5, top_n: int = 10) -> pd.Series:
    """
    Counts how often specific cast members appear in a user's highly-
    rated movies (VALIDATION-split only). Note: movies_master doesn't
    have a separate "director" field from Week 1 -- only `cast_list`
    (top-billed cast) was extracted. Directors would need a separate
    TMDB credits field (crew, not cast) we didn't pull in Week 1 --
    flagging this as a real gap, addressed below.
    """
    user_ratings = ratings_val[(ratings_val["userId"] == user_id) & (ratings_val["rating"] >= min_rating)]
    user_movies = movies_master[movies_master["movieId"].isin(user_ratings["movieId"])]

    cast_counts = {}
    for cast_list in user_movies["cast_list"]:
        for actor in cast_list:
            cast_counts[actor] = cast_counts.get(actor, 0) + 1

    return pd.Series(cast_counts).sort_values(ascending=False).head(top_n)

import ast

def extract_director(crew_str):
    """TMDB's crew field is a stringified list of dicts, each with a 'job' key.
    Director is the crew member with job == 'Director'."""
    if pd.isna(crew_str):
        return None
    try:
        crew = ast.literal_eval(crew_str)
        directors = [person["name"] for person in crew if person.get("job") == "Director"]
        return directors[0] if directors else None  # take the first credited director
    except (ValueError, SyntaxError):
        return None


def add_director_to_movies_master(movies_master: pd.DataFrame, credits_path: str) -> pd.DataFrame:
    """Merges director info into movies_master using tmdb_id -> credits.csv's id."""
    credits = pd.read_csv(credits_path)
    credits_dedup = credits.drop_duplicates(subset="id", keep="first")
    credits_dedup["director"] = credits_dedup["crew"].apply(extract_director)

    out = movies_master.merge(
        credits_dedup[["id", "director"]], left_on="tmdb_id", right_on="id", how="left"
    ).drop(columns=["id"])
    return out

def get_user_director_affinity(user_id: int, ratings_val: pd.DataFrame, movies_master_with_director: pd.DataFrame,
                                min_rating: float = 3.5, top_n: int = 10) -> pd.Series:
    user_ratings = ratings_val[(ratings_val["userId"] == user_id) & (ratings_val["rating"] >= min_rating)]
    user_movies = movies_master_with_director[movies_master_with_director["movieId"].isin(user_ratings["movieId"])]

    director_counts = user_movies["director"].dropna().value_counts()
    return director_counts.head(top_n)


def build_affinity_bar_chart(affinity_series: pd.Series, title: str):
    """Horizontal bar chart -- reads better than vertical for name labels,
    which can be long (e.g. 'Samuel L. Jackson')."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=affinity_series.index[::-1],  # reverse so highest count is at top
        x=affinity_series.values[::-1],
        orientation='h',
    ))
    fig.update_layout(title=title, xaxis_title="Movies Rated Highly")
    return fig

def get_user_taste_trajectory(user_id: int, ratings_val: pd.DataFrame, movies_master: pd.DataFrame,
                               min_rating: float = 3.5, n_periods: int = 5) -> pd.DataFrame:
    """
    Splits a user's VALIDATION-split highly-rated movies into n_periods
    chronological chunks (by rating timestamp), and computes the genre
    distribution within each chunk -- shows how taste has shifted over
    the user's rating history, not just a single aggregate snapshot.
    """
    user_ratings = ratings_val[(ratings_val["userId"] == user_id) & (ratings_val["rating"] >= min_rating)].copy()
    user_ratings = user_ratings.sort_values("timestamp")

    if len(user_ratings) < n_periods:
        n_periods = max(1, len(user_ratings) // 2)  # fall back gracefully for thin histories

    user_ratings["period"] = pd.qcut(range(len(user_ratings)), q=n_periods, labels=False, duplicates="drop")

    merged = user_ratings.merge(movies_master[["movieId", "genre_list", "ml_genre_list"]], on="movieId")

    trajectory_rows = []
    for period, group in merged.groupby("period"):
        genre_counts = {}
        for _, row in group.iterrows():
            genres = row["genre_list"] if len(row["genre_list"]) > 0 else row["ml_genre_list"]
            for g in genres:
                norm = normalize_genre_name(g)
                if norm:
                    genre_counts[norm] = genre_counts.get(norm, 0) + 1
        for genre, count in genre_counts.items():
            trajectory_rows.append({"period": period, "genre": genre, "count": count})

    return pd.DataFrame(trajectory_rows)


def build_trajectory_chart(trajectory_df: pd.DataFrame, top_n_genres: int = 5, title: str = "Taste Trajectory"):
    """Line chart, one line per top genre, showing count across periods."""
    top_genres = trajectory_df.groupby("genre")["count"].sum().sort_values(ascending=False).head(top_n_genres).index

    fig = go.Figure()
    for genre in top_genres:
        genre_data = trajectory_df[trajectory_df["genre"] == genre].sort_values("period")
        fig.add_trace(go.Scatter(
            x=genre_data["period"], y=genre_data["count"],
            mode='lines+markers', name=genre,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time Period (earliest → most recent)",
        yaxis_title="Movies Rated Highly",
    )
    return fig


if __name__ == "__main__":
    movies_master, ratings_val = load_dashboard_data()

    # print("ratings_val shape:", ratings_val.shape)
    # print("unique users in val:", ratings_val["userId"].nunique())

    # # Pick a user with a reasonable amount of val-split activity
    # user_counts = ratings_val.groupby("userId").size().sort_values(ascending=False)
    # test_user = user_counts.index[0]
    # print(f"\ntesting userId={test_user} ({user_counts.iloc[0]} val ratings)")

    # genre_profile = get_user_genre_profile(test_user, ratings_val, movies_master)
    # print("\ngenre profile:")
    # print(genre_profile)

    # test_user = 103611
    # genre_profile = get_user_genre_profile(test_user, ratings_val, movies_master)

    # fig = build_genre_radar_chart(genre_profile, top_n=8, title=f"Genre Preferences — User {test_user}")

    # # Save as HTML to view locally (since we're not in Streamlit yet)
    # fig.write_html("dashboard_test_radar.html")

    # decade_profile = get_user_decade_profile(test_user, ratings_val, movies_master)
    # print("decade profile:")
    # print(decade_profile)

    # fig = build_decade_bar_chart(decade_profile, title=f"Decade Preferences — User {test_user}")
    # fig.write_html("dashboard_test_decade.html")

    # cast_affinity = get_user_cast_affinity(test_user, ratings_val, movies_master)
    # print("top cast affinity:")
    # print(cast_affinity)

    # CREDITS_PATH = "./data/raw/the_movies_dataset/credits.csv"

    # movies_master, ratings_val = load_dashboard_data()
    # movies_master_with_director = add_director_to_movies_master(movies_master, CREDITS_PATH)

    # missing_director = movies_master_with_director["director"].isna().sum()
    # print(f"movies with director info: {len(movies_master_with_director) - missing_director} "
    #       f"({100*(1 - missing_director/len(movies_master_with_director)):.1f}%)")
    # print(movies_master_with_director[["title", "director"]].dropna().head())

    # # Save this enriched version so we don't need to redo this every time
    # movies_master_with_director.to_parquet("data/interim/movies_master_with_director.parquet", index=False)

    # movies_master_with_director = pd.read_parquet("data/interim/movies_master_with_director.parquet")
    # _, ratings_val = load_dashboard_data()

  
    # director_affinity = get_user_director_affinity(test_user, ratings_val, movies_master_with_director)
    # print("top director affinity:")
    # print(director_affinity)

    # fig = build_affinity_bar_chart(director_affinity, title=f"Director Affinity — User {test_user}")
    # fig.write_html("dashboard_test_director.html")
    # print("\nsaved dashboard_test_director.html")

    test_user = 103611
    trajectory_df = get_user_taste_trajectory(test_user, ratings_val, movies_master, n_periods=5)
    print("trajectory data:")
    print(trajectory_df.head(20))

    fig = build_trajectory_chart(trajectory_df, top_n_genres=5, title=f"Taste Trajectory — User {test_user}")
    fig.write_html("dashboard_test_trajectory.html")
    print("\nsaved dashboard_test_trajectory.html")

    # Quick check: are periods actually balanced, and do genres show real movement across periods?
    print("ratings per period:")
    print(trajectory_df.groupby("period")["count"].sum())

    print("\nDrama trend across periods (as an example):")
    print(trajectory_df[trajectory_df["genre"] == "Drama"].sort_values("period"))