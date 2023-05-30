import io
import os
import sys
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import uuid

import requests

def get_strings_of_element(soup, element):
  selected = soup.select_one(element)

  if selected is None:
    return None

  return ' '.join(selected.stripped_strings)

def wait_for_element_to_appear(page, element):
  while True:
    try:
      page.wait_for_selector(element)
      break
    except:
      print(f"Failed to get element {element}, retrying...")
  return True

def get_car_urls_of_page(page, page_number):
  page.goto(f'https://m.autoplius.lt/skelbimai/naudoti-automobiliai?qt=&qt_autocomplete=&page_nr={page_number}&category_id=2&make_id%5B97%5D=1318')
  wait_for_element_to_appear(page, '.list-items')
  content = page.content()
  soup = BeautifulSoup(content, 'html.parser')
  list_items = soup.select_one('.list-items')

  urls = [a['href'] for a in list_items.select('a')]
  regexp = re.compile(r"\d-\d-l")
  urls = [d for d in urls if regexp.search(d) is not None]
  urls = [d for d in urls if "skelbimai" in d]
  return [d for d in urls if not "page_nr" in d]

def get_car_data(page, url):
  try:
    page.goto(url)
    result = wait_for_element_to_appear(page, '.main-price')
    if result == False:
      return None, None
    content = page.content()
    soup = BeautifulSoup(content, 'html.parser')

    bigPhoto = soup.select_one('.bigphoto')
    if bigPhoto is None:
      return None, None
    photoSrc = soup.select_one('.bigphoto').find('source', attrs={'media': '(min-width: 401px)'}).get('srcset')
    _, file_extension = os.path.splitext(photoSrc)
    generated_uuid = uuid.uuid4()
    filename = f"{generated_uuid}{file_extension}"

    car = {
      "id": url,
      "title": get_strings_of_element(soup, '.title'),
      "price": get_strings_of_element(soup, '.main-price'),
      "make_date": get_strings_of_element(soup, '.field_make_date'),
      "kilometrage": get_strings_of_element(soup, '.field_kilometrage'),
      "engine": get_strings_of_element(soup, '.field_engine'),
      "fuel_id": get_strings_of_element(soup, '.field_fuel_id'),
      "body_type_id": get_strings_of_element(soup, '.field_fuel_id'),
      "number_of_doors": get_strings_of_element(soup, '.field_number_of_doors_id'),
      "wheel_drive": get_strings_of_element(soup, '.field_wheel_drive_id'),
      "gearbox": get_strings_of_element(soup, '.field_gearbox_id'),
      "condition_type": get_strings_of_element(soup, '.field_condition_type_id'),
      "color": get_strings_of_element(soup, '.field_color_id'),
      "damaged": get_strings_of_element(soup, '.field_has_damaged_id'),
      "steering_wheel": get_strings_of_element(soup, '.field_steering_wheel_id'),
      "mot_date": get_strings_of_element(soup, '.field_mot_date'),
      "wheel_radius": get_strings_of_element(soup, '.field_wheel_radius_id'),
      "weight": get_strings_of_element(soup, '.field_weight'),
      "number_of_seats": get_strings_of_element(soup, '.field_number_of_seats_id'),
      "origin_country": get_strings_of_element(soup, '.field_origin_country_id'),
      "euro_id": get_strings_of_element(soup, '.field_euro_id'),
      "co2": get_strings_of_element(soup, '.field_co2'),
      "fuel_cons_urban": get_strings_of_element(soup, '.field_fuel_cons_urban'),
      "fuel_cons_extra_urban": get_strings_of_element(soup, '.field_fuel_cons_extra_urban'),
      "fuel_cons_combined": get_strings_of_element(soup, '.field_fuel_cons_combined'),
      "image": filename
    }
    return car, photoSrc
  except:
    return None, None

if __name__ == '__main__':
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
  sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

  df = pd.DataFrame({})
  if os.path.exists('cars2.csv'):
    df = pd.read_csv('cars2.csv')

  with sync_playwright() as p:
    chrome_executable_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
    browser = p.chromium.launch(executable_path=chrome_executable_path, headless=False)
    page = browser.new_page()

    for page_nr in range(2, 160):
      print(f"Getting urls of page {page_nr}")
      urls = get_car_urls_of_page(page, page_nr)
      for url in urls:
        print(f"Getting car from url {url}")
        car, photoSrc = get_car_data(page, url)
        if car is None:
          continue
        if df.empty or not df['id'].isin([car['id']]).any():
          image_response = requests.get(photoSrc)
          filename = car['image']
          with open(f'data/images/{filename}', 'wb') as image_file:
            image_file.write(image_response.content)
          df = df.append(car, ignore_index=True)
          df.to_csv('cars2.csv', index=False)
    
    browser.close()