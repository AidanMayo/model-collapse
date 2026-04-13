import html
import math
import re
from collections import Counter

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
        top_words = word_counts.most_common(10)

        entropy = 0;
        for count in word_counts.values():
            p_i = count / total_words
            entropy -= p_i * math.log(p_i, 2)

        print(f"--- Analysis Results for {text_file} ---")
        print(f"Total Words:     {total_words:,}")
        print(f"Unique Words:    {unique_words:,}")
        print(f"Overall Entropy: {entropy:.4f} bits/word")
        print("\nTop 10 Most Frequent Words:")
        for word, count in top_words[:10]:
            print(f"{word:15}: {count:,}")
    except Exception as e:
       print(e)


if __name__ == '__main__':
    get_and_print_stats('80kOutput-0percent.txt')
