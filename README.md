# German Credit Risk Prediction

A complete end-to-end machine learning pipeline for predicting **credit risk** using structured customer data. This project includes data preprocessing, model training, evaluation, and deployment via a Streamlit web application.

The dataset has been sourced from this kaggle dataset: [German Credit Risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk/data).

## Requirements

- Python 3.8+
- `pyenv`
- `pip`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rahulehh/credit-risk-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
    cd credit-risk-prediction
   ```
3. Change your python version from .python-version file:
   ```bash
   pyenv local
   ```
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
6. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
7. Run program

   ```bash
   # To run the model build process:
   python main.py --run-build

   # To launch the Streamlit app:
   python main.py --run-app
   ```

### Google Colab

You can also build and run the model directly in Colab to explore the data, perform exploratory data analysis (EDA), and gain a deeper understanding of the dataset and the model pipeline.

<a href="https://colab.research.google.com/drive/1woSHfXaKZNPP-z0Bac8uKjffSGKMF1o0?usp=sharing">
<img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
</a>
