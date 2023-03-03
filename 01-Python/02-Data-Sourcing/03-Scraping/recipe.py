# pylint: disable=missing-docstring,line-too-long
import sys
from os import path
import csv
import requests
from bs4 import BeautifulSoup

BASE_URI = "https://recipes.lewagon.com/?search[query]="


def parse(html):
    ''' return a list of dict {name, difficulty, prep_time} '''
    soup = BeautifulSoup(html, "html.parser")
    zipp = zip( soup.find_all('p', class_='recipe-name'), 
                soup.find_all('span', class_='recipe-difficulty'), 
                soup.find_all('span', class_='recipe-cooktime'))
    lst = []
    for name, diff, time in zipp:
        print(name.text, diff.text, time.text)
        lst.append({'name': name.text, 'difficulty': diff.text, 'prep_time': time.text})
    return lst

def parse_recipe():
    ''' return a dict {name, difficulty, prep_time} modelising a recipe'''
    return {}

def write_csv(ingredient, recipes):
    ''' dump recipes to a CSV file `recipes/INGREDIENT.csv` '''
    file = open('recipes/'+ingredient+'.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(recipes)
    file.close()

def scrape_from_internet(ingredient, start=1):
    ''' Use `requests` to get the HTML page of search results for given ingredients. '''
    res = ''
    for i in range(1, start+1):
        res += str(requests.get(BASE_URI+ingredient+'&page='+str(i)).content)
    return res

def scrape_from_file(ingredient):
    file = f"pages/{ingredient}.html"
    if path.exists(file):
        return open(file)
    print("Please, run the following command first:")
    print(f'curl "https://recipes.lewagon.com/?search[query]={ingredient}" > pages/{ingredient}.html')
    sys.exit(1)


def main():
    if len(sys.argv) > 1:
        ingredient = sys.argv[1]
        recipes = parse(scrape_from_internet(ingredient))
        write_csv(ingredient, recipes)
    else:
        print('Usage: python recipe.py INGREDIENT')
        sys.exit(0)


if __name__ == '__main__':
    main()
