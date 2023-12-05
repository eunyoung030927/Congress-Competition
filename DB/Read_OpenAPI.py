import requests

url = '..'
params ={'serviceKey' : '', 'numOfRows' : '10', 'pageNo' : '1', 'class_code' : '1', 'commCode' : '' }

response = requests.get(url, params=params)
print(response.content)