import kagglehub


def download_dataset(dataset):
    dataset_path = kagglehub.dataset_download(dataset)
    print("Dataset download complete")
    return dataset_path


if __name__ == "__main__":
    download_dataset("https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk/data")
