import gc
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import re
import html
import os
from collections import Counter

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

def prepare_data_for_lstm(raw_review):
    cleaned = clean_text(raw_review)
    return f"<start> {cleaned} <end>"

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length, max_tokens=50000):
        print('Loading and cleaning data...')
        processed_words = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    cleaned = prepare_data_for_lstm(line)
                    processed_words.extend(cleaned.split())

        print(f"Total words (including markers): {len(processed_words)}")

        word_counts = Counter(processed_words)
        common_words = [w for w, _ in word_counts.most_common(max_tokens - 1)]
        self.vocab = {word: i + 1 for i, word in enumerate(common_words)}
        self.vocab['[UNK]'] = 0
        self.vocab_size = len(self.vocab)

        all_ids = [self.vocab.get(w, 0) for w in processed_words]

        self.x = []
        self.y = []
        for i in range(0, len(all_ids) - seq_length):
            self.x.append(all_ids[i : i + seq_length])
            self.y.append(all_ids[i + seq_length])

        print(f"Dataset created with {len(self.x)} sequences.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

class ReviewLSTM(nn.Module):
    def __init__(self, vocab_size, seq_length):
        super(ReviewLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm1 = nn.LSTM(256, 1024, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(1024, 1024, batch_first=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, (h_n, c_n) = self.lstm2(x)
        x = self.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x

def setup_and_train(file_path, save_name, seq_length=50, batch_size=1024, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TextDataset(file_path, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ReviewLSTM(dataset.vocab_size, seq_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    print('Training...')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'vocab': dataset.vocab,
        }, f'checkpoints/{save_name}_epoch_{epoch+1:02d}.pt')

    print('Training Done')
    return model

def generate_text(model, vocab, seed_text, seq_length, gen_length=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    int_to_word = {i: w for w, i in vocab.items()}

    words = seed_text.lower().split()
    if len(words) < seq_length:
        words = ['[UNK]'] * (seq_length - len(words)) + words

    generated = words

    with torch.no_grad():
        for _ in range(gen_length):
            current_seq = [vocab.get(w, 0) for w in generated[-seq_length:]]
            input_tensor = torch.tensor([current_seq]).to(device)

            output = model(input_tensor)

            logits = output / temperature
            probabilities = F.softmax(logits, dim=-1)

            next_id = torch.multinomial(probabilities, 1).item()

            generated.append(int_to_word.get(next_id, '[UNK]'))

    return " ".join(generated[len(words)-len(seed_text.split()):])

def generate_from_scratch(model, vocab, seq_length, stop_token="<end>", temperature=0.8, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    int_to_word = {i: w for w, i in vocab.items()}

    current_window = ["[UNK]"] * (seq_length - 1) + ["<start>"]

    generated_output = []

    with torch.no_grad():
        for _ in range(1000):
            input_ids = [vocab.get(w, 0) for w in current_window[-seq_length:]]
            input_tensor = torch.tensor([input_ids]).to(device)

            logits = model(input_tensor)

            logits = logits / temperature

            logits[0][0] = -float('Inf')

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            next_word = int_to_word.get(next_id, "[UNK]")

            # print(next_word, end=" ", flush=True)

            if next_word == stop_token:
                break

            if next_word not in ["[UNK]", "<start>", "<end>"]:
                generated_output.append(next_word)

            current_window.append(next_word)

    return " ".join(generated_output)

def generate_batch(model, vocab, seq_length, batch_size=64, temperature=0.8, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    int_to_word = {i: w for w, i in vocab.items()}

    start_id = vocab.get("<start>", 0)
    unk_id = vocab.get("[UNK]", 0)
    end_id = vocab.get("<end>", 0)

    current_batch = torch.full((batch_size, seq_length - 1), unk_id, dtype=torch.long).to(device)
    starts = torch.full((batch_size, 1), start_id, dtype=torch.long).to(device)
    current_batch = torch.cat([current_batch, starts], dim=1)

    finished_reviews = [[] for _ in range(batch_size)]
    is_done = [False] * batch_size

    with torch.no_grad():
        for _ in range(1000):
            logits = model(current_batch)
            logits = logits / temperature

            logits[:, unk_id] = -float('Inf')

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, 1)

            for i in range(batch_size):
                if not is_done[i]:
                    token_id = next_ids[i].item()
                    if token_id == end_id:
                        is_done[i] = True
                    else:
                        finished_reviews[i].append(int_to_word.get(token_id, "[UNK]"))

            if all(is_done):
                break

            current_batch = torch.cat([current_batch[:, 1:], next_ids], dim=1)

    return [" ".join(r) for r in finished_reviews]

SEQ_LEN = 50
BATCH_SIZE = 1024

if __name__ == '__main__':
    # st = (time.time())

    # trained_model = setup_and_train('80kOutput-0percent.txt', "0contamination")

    # checkpoint = torch.load('checkpoints/0contamination_epoch_25.pt')
    # vocab = checkpoint['vocab']

    # model = ReviewLSTM(len(vocab), seq_length=50).to("cuda")
    # model.load_state_dict(checkpoint['model_state_dict'])

    # created = 0

    # with open("output0-lstm-100k.txt", mode='w', newline='') as f:
    #     while created < 100000:
    #         reviews = generate_batch(model, vocab, seq_length=50, batch_size=64)

    #         for rev in reviews:
    #             if len(rev.split()) > 15:
    #                 f.write(rev + "\n")
    #                 created += 1

    # print(f"Total Generation 1 Creation Time: {(time.time()-st)/3600:.2f} hours")

    # del model
    # torch.cuda.empty_cache()

    sample_lines(20000, "output0-lstm-100k.txt", "20kOutput-lstm.txt")
    merge("20kOutput-lstm.txt", "60kOutput-normal.txt", "80kOutput-lstm-25percent.txt")

    st1 = (time.time())

    setup_and_train('80kOutput-lstm-25percent.txt', "25contamination")

    checkpoint1 = torch.load('checkpoints/25contamination_epoch_25.pt')
    vocab1 = checkpoint1['vocab']

    model1 = ReviewLSTM(len(vocab1), seq_length=50).to("cuda")
    model1.load_state_dict(checkpoint1['model_state_dict'])

    created1 = 0

    with open("output1-lstm-100k.txt", mode='w', newline='') as f:
        while created1 < 100000:
            reviews = generate_batch(model1, vocab1, seq_length=50, batch_size=64)

            for rev in reviews:
                if len(rev.split()) > 15:
                    f.write(rev + "\n")
                    created1 += 1

    print(f"Total Generation 1 Creation Time: {(time.time()-st1)/3600:.2f} hours")

    del model1
    gc.collect()
    torch.cuda.empty_cache()

    sample_lines(40000, "output1-lstm-100k.txt", "40kOutput-lstm.txt")
    merge("40kOutput-lstm.txt", "40kOutput-normal.txt", "80kOutput-lstm-50percent.txt")

    st2 = (time.time())

    setup_and_train('80kOutput-lstm-50percent.txt', "50contamination")

    checkpoint2 = torch.load('checkpoints/50contamination_epoch_25.pt')
    vocab2 = checkpoint2['vocab']

    model2 = ReviewLSTM(len(vocab2), seq_length=50).to("cuda")
    model2.load_state_dict(checkpoint2['model_state_dict'])

    created2 = 0

    with open("output2-lstm-100k.txt", mode='w', newline='') as f:
        while created2 < 100000:
            reviews = generate_batch(model2, vocab2, seq_length=50, batch_size=64)

            for rev in reviews:
                if len(rev.split()) > 15:
                    f.write(rev + "\n")
                    created2 += 1

    print(f"Total Generation 2 Creation Time: {(time.time()-st2)/3600:.2f} hours")

    del model2
    gc.collect()
    torch.cuda.empty_cache()

    sample_lines(60000, "output2-lstm-100k.txt", "60kOutput-lstm.txt")
    merge("60kOutput-lstm.txt", "20kOutput-normal.txt", "80kOutput-lstm-75percent.txt")

    st3 = (time.time())

    setup_and_train('80kOutput-lstm-75percent.txt', "75contamination")

    checkpoint3 = torch.load('checkpoints/75contamination_epoch_25.pt')
    vocab3 = checkpoint3['vocab']

    model3 = ReviewLSTM(len(vocab3), seq_length=50).to("cuda")
    model3.load_state_dict(checkpoint3['model_state_dict'])

    created3 = 0

    with open("output3-lstm-100k.txt", mode='w', newline='') as f:
        while created3 < 100000:
            reviews = generate_batch(model3, vocab3, seq_length=50, batch_size=64)

            for rev in reviews:
                if len(rev.split()) > 15:
                    f.write(rev + "\n")
                    created3 += 1

    print(f"Total Generation 3 Creation Time: {(time.time()-st3)/3600:.2f} hours")

    del model3
    gc.collect()
    torch.cuda.empty_cache()

    sample_lines(80000, "output3-lstm-100k.txt", "80kOutput-lstm-100percent.txt")

    st4 = (time.time())

    setup_and_train('80kOutput-lstm-100percent.txt', "100contamination")

    checkpoint4 = torch.load('checkpoints/100contamination_epoch_25.pt')
    vocab4 = checkpoint4['vocab']

    model4 = ReviewLSTM(len(vocab4), seq_length=50).to("cuda")
    model4.load_state_dict(checkpoint4['model_state_dict'])

    created4 = 0

    with open("output4-lstm-100k.txt", mode='w', newline='') as f:
        while created4 < 100000:
            reviews = generate_batch(model4, vocab4, seq_length=50, batch_size=64)

            for rev in reviews:
                if len(rev.split()) > 15:
                    f.write(rev + "\n")
                    created4 += 1

    print(f"Total Generation 4 Creation Time: {(time.time()-st4)/3600:.2f} hours")

    del model4
    gc.collect()
    torch.cuda.empty_cache()