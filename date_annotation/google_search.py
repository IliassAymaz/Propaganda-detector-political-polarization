import urllib
from bs4 import BeautifulSoup
import requests

class Search:
    pass
    
    def query_google(text):
        text = urllib.parse.quote_plus(text)
        url = 'https://google.com/search?hl=en&gl=en&q=' + text
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, 'lxml')
        

        # This identifies the div class that contains the query results
        queryDivs = soup.find_all("div", {"class" : "rc" })
        results = []
        for x in queryDivs:
            anotherSoup = BeautifulSoup(x.prettify(), 'lxml')
            titleResult = anotherSoup.find_all("span", class_ = "S3Uucc")
            dateResult = anotherSoup.find_all("span", class_ = "f")
            title = ""
            date = ""
            if len(titleResult) > 0:
                title = titleResult[0].text
            if len(dateResult) > 0 and len(titleResult) > 0:
                if "Rating" not in dateResult[0].text:
                    date = dateResult[0].text
            if title != "":
                results.append((' '.join(title.split()), ' '.join(date.split()).split(' -')[0]))
        return results
