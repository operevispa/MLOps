import requests

try:
    response = requests.get("http://backend:8000/get_features_info")
    print("Запрос http://backend:8000/get_features_info работает!")
    print(response.status_code)
except:
    print("ОШИБКА backend:8000")

try:
    response = requests.get("http://127.0.0.1:8000/get_features_info")
except:
    print("ОШИБКА 127.0.0.1:8000")

try:
    response = requests.get("http://0.0.0.0:8000/get_features_info")
except:
    print("ОШИБКА 0.0.0.0:8000")

try:
    response = requests.get("http://localhost:8000/get_features_info")
    # print(response.status_code)
except:
    print("ОШИБКА localhost:8000")
