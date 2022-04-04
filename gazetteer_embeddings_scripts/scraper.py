import requests
import re
from bs4 import BeautifulSoup
import io

urls = [
    "https://en.wikipedia.org/wiki/Comparison_of_integrated_development_environments"
    ]

for url in urls:
    text = []

    table_name = url.split("/")[4]

    fileName = table_name + '.txt'

    f = io.open(f'./code_gazetteers/{fileName}', 'w', encoding='utf8')

    s = requests.Session()
    response = s.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    all_tables = soup.find_all('table')
    lst_data = []
    for right_table in soup.find_all('table', {"class": 'wikitable sortable'}):
        rows = right_table.findAll("tr")
        for row in rows:
            data = [d.text.rstrip() for d in row.find_all('th')]
            lst_data.append(data)
            print(data)

    for item in lst_data:
        f.write("%s\n" % item)

    f.close()
