import requests
from bs4 import BeautifulSoup

links = []
base_url = "https://www.royalroad.com/fictions/best-rated?page="
NUMBER_OF_PAGES = 500

for page in range(1, NUMBER_OF_PAGES + 1):
    url = base_url + str(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for item in soup.find_all('div', class_='fiction-list-item row'):
        a_tag = item.find('a', href=True)
        if a_tag:
            links.append(a_tag['href'])

with open('output.txt', 'w') as f:
    for i, link in enumerate(links, 1):
        f.write(str(i) + ",https://www.royalroad.com" + link + '\n')

print(f"Extracted {len(links)} links to output.txt")
