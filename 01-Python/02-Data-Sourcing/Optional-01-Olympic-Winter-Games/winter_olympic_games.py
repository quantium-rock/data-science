# pylint: disable=missing-docstring

import csv

COUNTRIES_FILEPATH = "data/dictionary.csv"
MEDALS_FILEPATH = "data/winter.csv"

def read_files(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        lst = []
        for row in reader:
            lst.append(row)
        return lst

MEDALS = read_files(MEDALS_FILEPATH)
COUNTRIES = read_files(COUNTRIES_FILEPATH)

def most_decorated_athlete_ever():
    """Returns who won the most winter olympic games medals (gold/silver/bronze) ever"""
    atheletes = {}
    for lst in MEDALS:
        try: 
            atheletes[lst[4]] += 1
        except: 
            atheletes[lst[4]] = 1
    return max(atheletes, key=atheletes.get)


def country_with_most_gold_medals(min_year, max_year):
    """Returns which country won the most gold medals between `min_year` and `max_year`"""
    countries = {}
    for lst in MEDALS:
        if int(lst[0]) >= min_year and int(lst[0]) <= max_year and lst[8] == 'Gold':
            try: 
                countries[lst[5]] += 1
            except: 
                countries[lst[5]] = 1
    code = max(countries, key=countries.get)
    for lst in COUNTRIES:
        if lst[1] == code:
            return lst[0]
    

def top_three_women_in_five_thousand_meters():
    """Returns the three women with the most 5000 meters medals(gold/silver/bronze)"""
    women = {}
    for lst in MEDALS:
        if lst[6] == 'Women' and lst[7] == '5000M':
            try: 
                women[lst[4]] += 1
            except: 
                women[lst[4]] = 1
    return sorted(women, key=women.get, reverse=True)[:3]
