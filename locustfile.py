from locust import HttpUser, task, between

class WasteUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        with open("tests/sample.jpg", "rb") as f:
            self.img = f.read()

    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("sample.jpg", self.img, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")