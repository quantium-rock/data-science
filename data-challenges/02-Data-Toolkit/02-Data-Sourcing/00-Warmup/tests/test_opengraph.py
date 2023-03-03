# pylint: disable-all

import unittest
from opengraph import fetch_metadata

class TestOpenGraph(unittest.TestCase):
    def test_lewagon_com(self):
        data = fetch_metadata("https://www.lewagon.com")
        self.assertRegex(data["title"], r'(?i)le wagon')

    def test_a_com(self):
        data = fetch_metadata("https://www.a.com") # Does not exist
        self.assertTrue(data is None or data == "")

    def test_empty_url(self):
        data = fetch_metadata("")
        self.assertTrue(data is None or data == "")