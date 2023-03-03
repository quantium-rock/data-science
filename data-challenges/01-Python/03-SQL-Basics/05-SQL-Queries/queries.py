# pylint: disable=C0103, missing-docstring

def detailed_movies(db):
    '''return the list of movies with their genres and director name'''
    query = """ SELECT movies.title, movies.genres, directors.name FROM movies
                JOIN directors ON movies.director_id = directors.id """
    return db.execute(query).fetchall()


def late_released_movies(db):
    '''return the list of all movies released after their director death'''
    query = """ SELECT movies.title FROM movies
                JOIN directors ON movies.director_id = directors.id
                WHERE movies.start_year > directors.death_year 
                ORDER BY movies.title ASC """
    return [ x[0] for x in db.execute(query).fetchall() ]


def stats_on(db, genre_name):
    '''return a dict of stats for a given genre'''
    query = """ SELECT COUNT(movies.genres), ROUND(AVG(movies.minutes),2) FROM movies
                WHERE movies.genres = ? """
    out = db.execute(query, [genre_name]).fetchall()[0]
    return { 'genre': genre_name, 'number_of_movies': out[0], 'avg_length': out[1] }


def top_five_directors_for(db, genre_name):
    '''return the top 5 of the directors with the most movies for a given genre'''
    query = """ SELECT directors.name, COUNT(directors.name) FROM movies
                JOIN directors ON movies.director_id = directors.id
                WHERE movies.genres = ?
                GROUP BY directors.name 
                ORDER BY COUNT(directors.name) DESC, directors.name ASC
                LIMIT 5 """
    return db.execute(query, [genre_name]).fetchall()


def movie_duration_buckets(db):
    '''return the movie counts grouped by bucket of 30 min duration'''
    query = """ SELECT (movies.minutes/30+1)*30 AS thirty, COUNT(movies.minutes)
                FROM movies
                WHERE movies.minutes IS NOT NULL
                GROUP BY thirty
                ORDER BY thirty ASC """
    return db.execute(query).fetchall()


def top_five_youngest_newly_directors(db):
    '''return the top 5 youngest directors when they direct their first movie'''
    query = """ SELECT directors.name, start_year-directors.birth_year AS age FROM movies
                JOIN directors ON movies.director_id = directors.id
                WHERE directors.birth_year IS NOT NULL
                GROUP BY directors.name
                ORDER BY age ASC
                LIMIT 5 """
    return db.execute(query).fetchall()
                    
