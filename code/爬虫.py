from requests_html import HTMLSession
from requests_html import HTML
import re

"""
https://baike.baidu.com/error.html
"""

session = HTMLSession()
url = 'https://baike.baidu.com/item/陆永'
# url = 'https://baike.baidu.com/item/王芳就'
response = session.get(url)

print(response.url)

para_elem = response.html.find('.para')
str_res = ""
for value in para_elem:
    str_res += re.sub(r'(\r|\n)*','',value.text)
print(str_res)

# print(response == None)

banks = response.html.find('.polysemantList-wrapper')

data = banks[0]

persion_links = list(data.absolute_links)

# print(persion_links)

if len(persion_links) >0:
    data_1 = persion_links[0]

    session = HTMLSession()
    response_1 = session.get(data_1)

    para_elem = response_1.html.find('.para')

    str_res = ""
    for value in para_elem:
        str_res += re.sub(r'(\r|\n)*','',value.text)
    print(str_res)

    print(data_1)

