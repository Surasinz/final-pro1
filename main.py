import requests

url = "https://api.aiforthai.in.th/panyapradit-ocr"

files = {'file': open('captured_image_20240620-181210.jpg', 'rb')}

headers = {
    'Apikey': "QQAfpfak9Ot0HLeGklytNd5EJl9f4jaE",
}

response = requests.post(url, files=files, headers=headers)

print(response.content)
