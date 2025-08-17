# data_download.py

from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

os.makedirs("data", exist_ok=True)

api.dataset_download_files(
    "ajaxianazarenka/premier-league",
    path="data",
    unzip=True
)

print("✅ Downloaded and extracted to data/")
