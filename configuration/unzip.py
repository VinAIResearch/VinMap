import zipfile


with zipfile.ZipFile("uaic2022_training_data_update.zip", "r") as zip_ref:
    zip_ref.extractall(".")
