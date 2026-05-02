import html
import math
import random
import re
from collections import Counter

def get_random_lines(count, input_file, output_file):
    with open(input_file, mode='r', newline='') as f:
        lines = f.readlines()
    sample_size = min (count, len(lines))
    subset = random.sample(lines, sample_size)
    with open(output_file, mode='w', newline='') as f:
        f.writelines(subset)

def clean_text(text):
    text = html.unescape(text)

    text = re.sub(r"<[^>]*>", " ", text)

    text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text

def get_and_print_stats(text_file):
    word_counts = Counter()
    total_words = 0

    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_text = clean_text(line)
                # print(cleaned_text)
                words = re.findall(r'\b\w+\b', cleaned_text.lower())

                word_counts.update(words)
                total_words += len(words)

        if total_words == 0:
            print("No words found")
            return

        unique_words = len(word_counts)
        top_words = word_counts.most_common(20)

        entropy = 0;
        for count in word_counts.values():
            p_i = count / total_words
            entropy -= p_i * math.log(p_i, 2)

        print(f"--- Analysis Results for {text_file} ---")
        print(f"Total Words:     {total_words:,}")
        print(f"Unique Words:    {unique_words:,}")
        print(f"Overall Entropy: {entropy:.4f} bits/word")
        print("\nTop 20 Most Frequent Words:")
        for word, count in top_words[:20]:
            print(f"{word:15}: {count:,} | {count/total_words:.4f} Set Proportion")
    except Exception as e:
       print(e)


if __name__ == '__main__':
    get_and_print_stats('80kOutput-0percent.txt')
    get_and_print_stats('global_corpus.txt')
    # get_random_lines(25, "80kOutput-0percent.txt", "random-real-for-lstm")
    # get_random_lines(5, "output0-lstm-100k.txt", "random-fake-lstm-gen0")
    # get_random_lines(5, "output1-lstm-100k.txt", "random-fake-lstm-gen1")
    # get_random_lines(5, "output2-lstm-100k.txt", "random-fake-lstm-gen2")
    # get_random_lines(5, "output3-lstm-100k.txt", "random-fake-lstm-gen3")
    # get_random_lines(5, "output4-lstm-100k.txt", "random-fake-lstm-gen4")
    # get_random_lines(5, "output1-100k.txt", "random-fake-gen1")
    # get_random_lines(5, "output2-100k.txt", "random-fake-gen2")
    # get_random_lines(5, "output3-100k.txt", "random-fake-gen3")
    # get_random_lines(5, "output4-100k.txt", "random-fake-gen4")
    get_and_print_stats('output0-100k.txt')
    get_and_print_stats('output1-100k.txt')
    get_and_print_stats('output2-100k.txt')
    get_and_print_stats('output3-100k.txt')
    get_and_print_stats('output4-100k.txt')
    get_and_print_stats('output0-lstm-100k.txt')
    get_and_print_stats('output1-lstm-100k.txt')
    get_and_print_stats('output2-lstm-100k.txt')
    get_and_print_stats('output3-lstm-100k.txt')
    get_and_print_stats('output4-lstm-100k.txt')
