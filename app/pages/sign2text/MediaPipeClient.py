import requests

class MediaPipeClient:
    def __init__(self, service_url):
        self.service_url = service_url
    
    def ping_sl2t(self):
        response = requests.post(f"{self.service_url}/process",
                             json={"value": True})
        print("STATUS:", response.status_code)
        print("HEADERS:", response.headers.get('Content-Type'))
        print("BODY:", repr(response.text))
        response.raise_for_status()
        return response.json()
