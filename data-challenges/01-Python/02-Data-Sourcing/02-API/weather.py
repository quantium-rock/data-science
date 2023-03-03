# pylint: disable=missing-module-docstring

import sys
import requests

BASE_URI = "https://weather.lewagon.com/"

def search_city(query):
    '''Look for a given city. If multiple options are returned, have the user choose between them.
       Return one city (or None)
    '''
    res = requests.get(BASE_URI+'geo/1.0/direct', params={'q': query}).json()
    if res:
        res = res[0]
        return {'name':res['name'], 'country':res['country'], 'lat':res['lat'], 'lon':res['lon']}
    return None

def weather_forecast(lat, lon):
    '''Return a 5-day weather forecast for the city, given its latitude and longitude.'''
    res = requests.get(BASE_URI+'data/2.5/forecast', params={'lat':lat, 'lon':lon}).json()
    lst = []
    for i in range(0, 8*5, 8):
        lst.append(res['list'][i])
    return lst

query='barcelona'
def main():
    '''Ask user for a city and display weather forecast'''
    city = {}
    while True:
        query = input("City?\n> ")
        city = search_city(query)
        if city:
            break
    weather = weather_forecast(city['lat'], city['lon'])
    print(weather[0]['weather']['main'])

if __name__ == '__main__':
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)
