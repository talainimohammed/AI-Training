import gensim.downloader
from bs4 import BeautifulSoup
import requests

def get_google_img(query):

    url = "https://www.google.com/search?q=" + str(query) + "&source=lnms&tbm=isch"
    headers={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

    html = requests.get(url, headers=headers).text

    soup = BeautifulSoup(html, 'html.parser')
    image = soup.find("img",{"class":"yWs4tf"})

    return image['src']
def html_page(title,l):
    strTable = "<html><h1>Similar Words For "+title+"</h1><table><tr><th>Keyword</th><th>Link</th></tr>"
    
    for n in l:
        strRW = "<tr><td>"+n+ "</td><td><img src="+l[n]+" alt="+n+" width='100' height='100'></td></tr>"
        strTable = strTable+strRW
    
    strTable = strTable+"</table></html>"
    
    hs = open("index.html", 'w')
    hs.write(strTable)
    return "Page Created"

glove_vector=gensim.downloader.load('glove-twitter-50')
mainword='twitter'
words=glove_vector.most_similar(mainword, topn=3)
d={}
for w in words:
    d[w[0]]=get_google_img(w[0])

print(html_page(mainword,d))
