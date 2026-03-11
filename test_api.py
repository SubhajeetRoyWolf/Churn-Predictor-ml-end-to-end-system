import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "price": 200,
    "freight_value": 40,
    "delivery_time": 7
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)

if response.status_code == 200:
    print("JSON Output:", response.json())
else:
    print("API Error")