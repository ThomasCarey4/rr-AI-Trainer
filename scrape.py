import requests
from bs4 import BeautifulSoup

with open('output.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

with open('scraped_output2.txt', 'w', encoding='utf-8') as out_file:
    for line in lines:
        id, url = line.split(',', 1)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ""
        title_container = soup.find('h1', class_='font-white')
        if title_container:
            title = title_container.get_text(strip=True)
        
        # Extract tags
        tags = []
        tags_container = soup.find('span', class_='tags')
        if tags_container:
            for a_tag in tags_container.find_all('a', class_='fiction-tag'):
                tags.append(a_tag.get_text(strip=True))

        # Extract blurb
        blurb = ""
        blurb_container = soup.find('div', class_='description')
        if blurb_container:
            hidden_content = blurb_container.find('div', class_='hidden-content')
            if hidden_content:
                blurb = "\n".join(p.get_text(strip=True) for p in hidden_content.find_all('p'))

        # Write to file with numbering
        if blurb or tags:
            out_file.write(f"<ENTRY id='{id}'>\n")
            out_file.write(f"<URL>{url}</URL>\n")
            out_file.write(f"<TITLE>{title}</TITLE>\n")
            out_file.write(f"<TAGS>{', '.join(tags)}</TAGS>\n")
            out_file.write(f"<BLURB>{blurb}</BLURB>\n")
            out_file.write("</ENTRY>\n\n")


print("Scraping complete. Data saved to scraped_output.txt.")

