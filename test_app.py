import requests

url = 'http://localhost:5001/predict'
try:
    with open("test_images/30424.jpg", 'rb') as img_file:
        files = {'image': ('30424.jpg', img_file, 'image/jpeg')}
        response = requests.post(url, files=files)
        print("Status Code:", response.status_code)
        print("Response Content:", response.text)
except Exception as e:
    print(f"Error: {e}")