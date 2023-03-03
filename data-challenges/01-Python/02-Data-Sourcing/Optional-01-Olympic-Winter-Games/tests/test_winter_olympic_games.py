# pylint: disable-all

import unittest

from winter_olympic_games import most_decorated_athlete_ever
from winter_olympic_games import country_with_most_gold_medals
from winter_olympic_games import top_three_women_in_five_thousand_meters

class TestWinterOlympicGames(unittest.TestCase):
    def test_most_decorated_athlete_ever(self):
        athlete = most_decorated_athlete_ever()
        self.assertEqual(athlete, 'BJOERNDALEN, Ole Einar')

    def test_country_with_most_gold_medals_between_2002_and_2014(self):
        country = country_with_most_gold_medals(2002, 2014)
        self.assertEqual(country, 'Canada')

    def test_country_with_most_gold_medals_between_1994_and_1998(self):
        country = country_with_most_gold_medals(1994, 1998)
        self.assertEqual(country, 'Germany')

    def test_top_three_women_in_five_thousand_meters(self):
        women = top_three_women_in_five_thousand_meters()
        self.assertEqual(women, ['PECHSTEIN, Claudia', 'NIEMANN-STIRNEMANN, Gunda', 'HUGHES, Clara'])
