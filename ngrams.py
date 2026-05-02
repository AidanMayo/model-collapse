import html
import pickle
import random
import re

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.lm import KneserNeyInterpolated, MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

import pandas as pd

# import kagglehub
# path = kagglehub.dataset_download("amazon-fine-food-reviews")

n = 3

# nltk.download('punkt')
# nltk.download('punkt_tab')

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

def sample_lines(count, input_file, output_file):
    with open(input_file, mode='r', newline='') as f:
        lines = f.readlines()
    sample_size = min (count, len(lines))
    subset = random.sample(lines, sample_size)
    with open(output_file, mode='w', newline='') as f:
        f.writelines(subset)

def merge(f1, f2, save):
    with open(f1, mode='r', newline='') as f:
        lines1 = f.readlines()

    with open(f2, mode='r', newline='') as f:
        lines2 = f.readlines()

    with open(save, mode='w', newline='') as f:
        f.writelines(lines1)
        f.writelines(lines2)

def clean_text(text):
    text = html.unescape(text)

    text = re.sub(r"<[^>]*>", " ", text)

    text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text

def train_LM(path_to_train_file):
    try:
        with open(path_to_train_file, 'r', encoding='utf-8') as f:
            print("Generating n-grams...")
            raw_text = f.read()

            sentences = sent_tokenize(raw_text.lower())
            tokens = [word_tokenize(clean_text(sent)) for sent in sentences]


            padded, data = padded_everygram_pipeline(n, tokens)

            print("Fitting model (this may hang for a minute)...")

            model = MLE(n)
            model.fit(padded, data)

            return model
    except Exception as e:
        print(e)

def LM_generate(LM_model, prompt):
    if prompt == "" or prompt is None:
        # print("Empty Prompt")
        context = []
    else:
        # print("Prompt: ", prompt)
        tokens = word_tokenize(prompt.lower())
        context = tokens[-(n-1):]

    # print("Beginning Generation...")
    # print(f"DEBUG: Context being sent to model: {context}")
    generated_tokens = []
    for _ in range(50):
        generation = LM_model.generate(1, text_seed=context)
        # print(generation)
        if generation == '</s>':
            break
        if generation != '<s>':
            generated_tokens.append(generation)
        context = (list(context) + [generation])[-2:]
    # print(f"Generated")

    result = " ".join(generated_tokens)
    final = (prompt + " " + result).strip()
    return final


if __name__ == '__main__':
    # grab_data(20000, "Reviews.csv", "20kOutput-normal.txt")
    # sample_lines(80000, "output3-100k.txt", "80kOutput-100percent.txt")
    merge("20kOutput-normal.txt", "60kOutput-normal.txt", "test.txt")
    foodLM1 = train_LM('80kOutput-100percent.txt')
    print("Loaded Model")

    created = 0

    with open("output4-100k.txt", mode='w', newline='') as f:
        while created < 100000:
            generation = LM_generate(foodLM1, prompt="")

            if len(generation.split(" ")) > 15:
                f.write(generation + "\n")
                created = created+1

