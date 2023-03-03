# pylint: disable=missing-docstring,invalid-name

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from bs4 import BeautifulSoup
import csv

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://recipes.lewagon.com/recipes/advanced")

search_input = driver.find_element_by_id('search_query') 
search_input.send_keys('chocolate')
search_input.submit()

wait = WebDriverWait(driver, 15)
wait.until(ec.visibility_of_element_located((By.XPATH, "//div[@id='recipes']")))

recipe_urls = []
cards = driver.find_elements_by_xpath("//div[@class='recipe my-3']")
print(f"Found {len(cards)} results on the page")
for card in cards:
    url = card.get_attribute('data-href')
    recipe_urls.append(url)

print(recipe_urls)


recipes = []
for url in recipe_urls:
    print(f"Navigating to {url}")
    driver.get(url)
    wait.until(ec.visibility_of_element_located((By.XPATH, "//div[@class='p-3 border bg-white rounded-lg recipe-container']")))

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    name = soup.find('h2').string.strip()
    cooktime = soup.find('span', class_='recipe-cooktime').text.strip()
    difficulty = soup.find('span', class_='recipe-difficulty').text.strip()
    price = soup.find('small', class_='recipe-price').attrs.get('data-price').strip()
    description = soup.find('p', class_='recipe-description').text.strip()
    recipes.append({
    'name': name,
    'cooktime': cooktime,
    'difficulty': difficulty,
    'price': price,
    'description': description
    })


with open('data/recipes.csv', 'w') as file:
  writer = csv.DictWriter(file, fieldnames=recipes[0].keys())
  writer.writeheader()
  writer.writerows(recipes)

driver.quit()

from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
