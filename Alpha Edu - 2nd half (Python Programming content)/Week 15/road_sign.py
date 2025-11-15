from inference_sdk import InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="pPEB7ATTlfxBxZHgBi7O"
)
result = CLIENT.infer("img.png", model_id="traffic_signs-rxmkx/1")

print(result["predictions"][0])