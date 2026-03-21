import html
import pickle
import re

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

import pandas as pd

# import kagglehub
# path = kagglehub.dataset_download("amazon-fine-food-reviews")

n = 5

nltk.download('punkt')
nltk.download('punkt_tab')

def grab_data(count, input_file, output_file):
    try:
        df = pd.read_csv(input_file, usecols=["Text"])

        # safety check just in case
        sample_size = min (count, len(df))
        subset = df["Text"].dropna().sample(n=sample_size)

        with open(output_file, mode='w', newline='') as f:
            for review in subset:
                f.write(review.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip() + "\n")

        print(f"Saved {subset.size} reviews to {output_file}")

    except Exception as e:
        print(e)

def clean_text(text):
    text = html.unescape(text)

    text = re.sub(r"<[^>]>", " ", text)

    text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text

def train_LM(path_to_train_file, save_name):
    try:
        with open(path_to_train_file, 'r', encoding='utf-8') as f:
            print("Generating n-grams...")
            raw_text = f.read()

            sentences = sent_tokenize(raw_text.lower())
            tokens = [word_tokenize(clean_text(sent)) for sent in sentences]


            padded, data = padded_everygram_pipeline(n, tokens)

            print("Fitting model (this may hang for a minute)...")

            model = KneserNeyInterpolated(n)
            model.fit(padded, data)

            with open(save_name, "wb") as f:
                pickle.dump(model, f)

            return model
    except Exception as e:
        print(e)

def load_LM(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model

def LM_generate(LM_model, prompt):
    sentences = sent_tokenize(prompt.lower())
    tokens = [word_tokenize(sent) for sent in sentences]

    context = tokens[-(n-1):]

    prediction = LM_model.generate(15, text_seed=context)

    clean_prediction = [word for word in prediction if word not in ["<s>", "</s>"]]

    print(prompt + " " + " ".join(clean_prediction))


if __name__ == '__main__':
    # grab_data(80000, "Reviews.csv", "80kOutput-0percent.txt")
    foodLM1 = train_LM('80kOutput-0percent.txt', 'lm0per')
    for i in range(5):
        LM_generate(foodLM1, prompt='')
