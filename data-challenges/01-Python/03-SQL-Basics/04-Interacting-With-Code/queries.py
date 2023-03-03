# pylint: disable=missing-docstring, C0103

def directors_count(db):
    # return the number of directors contained in the database
    return db.execute("SELECT COUNT(name) FROM directors ").fetchall()[0][0]


def directors_list(db):
    # return the list of all the directors sorted in alphabetical order
   return [ x[0] for x in db.execute("SELECT name FROM directors ORDER BY directors.name ASC").fetchall() ]


def love_movies(db):
    # return the list of all movies which contain the exact word "love"
    # in their title, sorted in alphabetical order
    query = """ SELECT title FROM movies
                WHERE movies.title LIKE 'love %' 
                OR movies.title LIKE '% love' 
                OR movies.title LIKE '% love %' 
                OR movies.title LIKE 'love,%' 
                OR movies.title LIKE '% love.%'
                OR movies.title LIKE "% love'%" 
                OR movies.title LIKE 'love'
                ORDER BY movies.title ASC """
    return [ x[0] for x in db.execute(query).fetchall() ]


def directors_named_like_count(db, name):
    # return the number of directors which contain a given word in their name
    query = """ SELECT COUNT(name) FROM directors
                WHERE directors.name LIKE ? """
    return db.execute(query, [f'%{name}%']).fetchone()[0]


def movies_longer_than(db, min_length):
    # return this list of all movies which are longer than a given duration,
    # sorted in the alphabetical order
    query = """ SELECT title, minutes FROM movies
                WHERE movies.minutes > ?
                ORDER BY movies.title ASC """
    return [ x[0] for x in db.execute(query, [min_length]).fetchall() ]
