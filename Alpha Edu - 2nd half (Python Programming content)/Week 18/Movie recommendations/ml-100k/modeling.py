import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Загрузка датасета рейтингов пользователей к фильмам
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('u.data', sep='\t', names=ratings_cols, encoding='latin-1')
print("Структура данных рейтингов пользователей:")
print(ratings_df.head())  # Логирование структуры данных

# Загрузка информации о фильмах
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
               'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies_df = pd.read_csv('u.item', sep='|', names=movies_cols, encoding='latin-1', usecols=['movie_id', 'title'])
print("Структура данных о фильмах:")
print(movies_df.head())  # Логирование структуры данных

# Объединяем датасет рейтингов с названиями фильмов
ratings_df = ratings_df.merge(movies_df, on='movie_id')
print("Структура объединенных данных:")
print(ratings_df.head())

# Создаем матрицу пользователей и фильмов (таблица user-item)
user_movie_matrix = ratings_df.pivot_table(index='user_id', columns='title', values='rating')
print("Фрагмент матрицы пользователь-фильм:")
print(user_movie_matrix.head())

# Заполняем пропущенные значения нулями (не просмотренные фильмы)
user_movie_matrix.fillna(0, inplace=True)

# Вычисляем косинусное сходство между пользователями
# Косинусное сходство измеряет угол между двумя векторами в многомерном пространстве.
# В данном случае, каждый пользователь представлен вектором его оценок фильмов.
# Чем меньше угол между двумя векторами, тем больше схожесть пользователей.
user_similarity = cosine_similarity(user_movie_matrix)

# Конвертируем в DataFrame для удобства работы
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
print("Фрагмент матрицы сходства пользователей:")
print(user_similarity_df.head())


# Функция для рекомендации фильмов на основе схожести пользователей
def recommend_movies(user_id, num_recommendations=5):
    """
    Рекомендует фильмы для пользователя на основе схожести с другими пользователями.

    Алгоритм:
    1. Найти пользователей, наиболее похожих на данного пользователя
    2. Посмотреть, какие фильмы они смотрели, но данный пользователь нет
    3. Выбрать фильмы с наивысшими средними рейтингами от схожих пользователей

    :param user_id: ID пользователя, для которого строится рекомендация
    :param num_recommendations: Количество рекомендуемых фильмов
    :return: Список рекомендованных фильмов
    """
    # Получаем индекс пользователя
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    print(f"Похожие пользователи для {user_id}: {similar_users.index.tolist()}")

    # Получаем фильмы, которые смотрели похожие пользователи
    movies_watched_by_user = user_movie_matrix.loc[user_id]
    recommended_movies = pd.Series(dtype='float64')

    for similar_user in similar_users.index:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        similar_user_unwatched = similar_user_ratings[movies_watched_by_user == 0]
        recommended_movies = pd.concat([recommended_movies, similar_user_unwatched])

    # Агрегируем оценки и сортируем
    recommended_movies = recommended_movies.groupby(recommended_movies.index).mean()
    recommended_movies = recommended_movies.sort_values(ascending=False).head(num_recommendations)

    print("Рекомендованные фильмы:")
    print(recommended_movies)

    return recommended_movies.index.tolist()


# Пример вызова функции (рекомендации для пользователя с ID 1)
recommended = recommend_movies(1, 5)
print("Рекомендуемые фильмы:", recommended)