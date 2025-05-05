import requests
import time


def request_predict_batch():
    return requests.post(
        url=f"http://127.0.0.1:5000/predict"
    )



if __name__ == '__main__':
    # We are calling the api each day
    while True:
        print("making request from the user")
        print(request_predict_batch().json())

        time.sleep(60*60*24)
