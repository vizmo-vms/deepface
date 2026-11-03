"""Checks that two running DeepFace API containers return the same results.

Sends the requests the Vizmo server makes (/represent with Facenet512, /verify with ArcFace)
to both containers and fails if any embedding, confidence or distance differs by more than
TOLERANCE.

Usage: python3 compare_images.py CURRENT_URL NEW_URL DATASET_DIR
"""

import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOLERANCE = 1e-4
REQUEST_TIMEOUT_SECONDS = 600


def wait_until_up(url):
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(3)
    sys.exit(f"{url} did not answer within 300 seconds")


def post(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"POST {url} returned {error.code}: {error.read().decode()[:500]}")


def data_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def max_difference(current, new):
    if len(current) != len(new):
        return float("inf")
    return max(abs(a - b) for a, b in zip(current, new))


def main():
    current_url, new_url, dataset = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    for url in (current_url, new_url):
        wait_until_up(url)

    failures = []

    def check(label, difference):
        status = "ok" if difference <= TOLERANCE else "DIFFERENT"
        print(f"{status:9} {label}: max difference {difference:.3g}")
        if difference > TOLERANCE:
            failures.append(label)

    represent_cases = [
        ("img1.jpg", "yunet"),
        ("img2.jpg", "yunet"),
        ("img3.jpg", "yunet"),
        ("couple.jpg", "yunet"),
        ("img1.jpg", "retinaface"),
    ]
    for image, detector in represent_cases:
        body = {
            "img": data_url(dataset / image),
            "model_name": "Facenet512",
            "detector_backend": detector,
            "enforce_detection": True,
            "max_faces": 1,
        }
        current = post(f"{current_url}/represent", body)["results"][0]
        new = post(f"{new_url}/represent", body)["results"][0]
        label = f"represent {image} {detector}"
        check(f"{label} embedding", max_difference(current["embedding"], new["embedding"]))
        check(
            f"{label} face_confidence",
            abs(current["face_confidence"] - new["face_confidence"]),
        )
        if current["facial_area"] != new["facial_area"]:
            print(f"DIFFERENT {label} facial_area: {current['facial_area']} != {new['facial_area']}")
            failures.append(f"{label} facial_area")

    verify_cases = [("img1.jpg", "img2.jpg"), ("img1.jpg", "img3.jpg")]
    for first, second in verify_cases:
        body = {
            "img1": data_url(dataset / first),
            "img2": data_url(dataset / second),
            "model_name": "ArcFace",
            "detector_backend": "yunet",
            "enforce_detection": True,
        }
        current = post(f"{current_url}/verify", body)
        new = post(f"{new_url}/verify", body)
        label = f"verify {first} {second} ArcFace"
        check(f"{label} distance", abs(current["distance"] - new["distance"]))
        if current["verified"] != new["verified"]:
            failures.append(f"{label} verified")

    if failures:
        sys.exit(f"{len(failures)} results differ: {', '.join(failures)}")
    print("Both images return the same results.")


if __name__ == "__main__":
    main()
