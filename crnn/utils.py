import torch
import cv2
import torchvision
import json
import Levenshtein


def levenshtetin(x: str):
    with open('data/word_bag.json') as f:
        bow = json.load(f)

    words = x.split(' ')
    processed_words = []

    for word in words:
        word = ''.join(filter(lambda x: str.isdigit(x) or str.isalpha(x), word))
        if word == '':
            continue
        word = word.lower()
        distance = [Levenshtein.distance(word, bag_word) for bag_word in bow]
        closest = distance.index(min(distance))
        if distance[closest] > (len(bow[closest]) // 2):
            processed_words.append(word)
        else:
            processed_words.append(bow[closest])

    return ' '.join(processed_words)


characters = ' !"#$%&\'()*+-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`lr{|}~·'


enc_dict = {}
for i, char in enumerate(characters):
    enc_dict[char] = i + 1


def encode(text):
    if isinstance(text, str):
        code = [enc_dict[c] for c in text]
        code = torch.tensor(code, dtype=torch.int32)
        return code
    elif isinstance(text, list):
        lengths = [len(t) for t in text]
        codes = []
        for t in text:
            codes.extend([enc_dict[c] for c in t])

        codes = torch.tensor(codes, dtype=torch.int32)
        lengths = torch.tensor(lengths, dtype=torch.int32)

        return codes, lengths


def decode(x):
    x = x.squeeze(1)
    x = torch.argmax(x, dim=1)
    text = []

    pc = -1
    for c in x:
        c = c.item()
        if c == pc or c == 0:
            continue
        else:
            text.append(c)
        pc = c

    text = [characters[c - 1] for c in text]
    text = ''.join(text)

    return text


def ocr(model, x):
    h, w = x.shape
    scale = 32 / h
    x = cv2.resize(x, dsize=None, fx=scale, fy=scale)
    x = torchvision.transforms.ToTensor()(x).unsqueeze(0)
    with torch.no_grad():
        out = model(x, torch.tensor(x.shape[3:], dtype=torch.int64))[0]
    text = decode(out)

    text = levenshtetin(text)

    return text
