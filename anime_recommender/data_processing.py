import pandas as pd

# Import datasets
UsersDF = pd.read_csv('../raw_data/users_cleaned.csv')
AnimesDF = pd.read_csv('../raw_data/anime_cleaned.csv')
ScoresDF = pd.read_csv('../raw_data/animelists_cleaned.csv')

# Trim ScoresDF to essential columns
ScoresDF = ScoresDF[['username', 'anime_id', 'my_score', 'my_status']]

# Find and print the range of ratings
lower_rating = ScoresDF['my_score'].min()
upper_rating = ScoresDF['my_score'].max()
print('Range of ratings vary between: {0} to {1}'.format(lower_rating, upper_rating))

# Generate a dataframe detailing number of animes rated by each user
user_anime_counts = ScoresDF['username'].value_counts()
UsersAndScores = pd.DataFrame({'username': user_anime_counts.index, 'animes_rated': user_anime_counts.values})

# Merge sampled user data with UsersAndScores
UsersSampled = UsersDF.sample(frac = 1) # Full sampling
UsersAndScoresSampled = pd.merge(UsersAndScores, UsersSampled, on = 'username')

# Group users by the same number of animes rated
UserRatedsAggregated = UsersAndScoresSampled['animes_rated'].value_counts().reset_index()
UserRatedsAggregated.columns = ['animes_rated', 'group_size']
UserRatedsAggregated.sort_values(by=['animes_rated'], inplace=True)

# Generate a dataframe detailing number of users rating each anime
anime_user_counts = ScoresDF['anime_id'].value_counts()
RatedsPerAnime = pd.DataFrame({'anime_id': anime_user_counts.index, 'number_of_users': anime_user_counts.values})

# Group animes by the same number of users rated
AnimeRatedsAggregated = RatedsPerAnime['number_of_users'].value_counts().reset_index()
AnimeRatedsAggregated.columns = ['number_of_users', 'group_size']
AnimeRatedsAggregated.sort_values(by=['number_of_users'], inplace=True)

# Filter out users and animes with less than 10 ratings
UserRatedsCutten = UsersAndScoresSampled[UsersAndScoresSampled['animes_rated'] >= 10]
AnimeRatedsCutten = RatedsPerAnime[RatedsPerAnime['number_of_users'] >= 10]

# Merge to create final dataset of users and animes with sufficient interactions
ScoresDFHotStart = pd.merge(ScoresDF, UserRatedsCutten, on = 'username')
ScoresDFHotStart = pd.merge(ScoresDFHotStart, AnimeRatedsCutten, on = 'anime_id')