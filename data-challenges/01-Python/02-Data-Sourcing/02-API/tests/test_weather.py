import datetime
import unittest
import weather


class TestWeather(unittest.TestCase):
    def test_search_city_for_paris(self):
        city = weather.search_city('Paris')
        self.assertEqual(city['name'], 'Paris')
        self.assertAlmostEqual(city['lat'], 48.858, 1)
        self.assertAlmostEqual(city['lon'], 2.3200, 1)

    def test_search_city_for_london(self):
        city = weather.search_city('London')
        self.assertEqual(city['name'], 'London')
        self.assertAlmostEqual(city['lat'], 51.507, 1)
        self.assertAlmostEqual(city['lon'], -0.127, 1)

    def test_search_city_for_unknown_city(self):
        city = weather.search_city('LGTM')
        self.assertEqual(city, None)

    def test_search_city_ambiguous_city(self):
        weather.input = lambda _: "0"
        city = weather.search_city('San')
        self.assertEqual(city['name'], 'Lhasa')

    def test_weather_forecast(self):
        forecast = weather.weather_forecast(51.5073219, -0.1276474)
        self.assertIsInstance(forecast, list, "Did you select the `consolidated_weather` key?")
        self.assertTrue(forecast[0].get('weather'))
