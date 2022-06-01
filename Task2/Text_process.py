import re
import numpy as np

def read_corpus(filename):
    # r = re.compile("[а-яА-Я .!,;:]+")
    lines = []
    with open(filename, 'r', encoding='Windows-1251', errors='ignore') as f:
        for pos, line in enumerate(f):
            # line = line.replace("\t", "").replace("\n", " ")
            # line = ''.join([c for c in filter(r.match, line)]) # оставить русские буквы
            #
            line = re.sub('[^а-яА-Я .!,;:]+', ' ', line.replace("\t", "").replace("\n", " ")).strip().lower()
            line = re.sub(" +", " ", line) # схлопываем пробелы
            line = line.replace(" .", ".")
            line = line.replace(" ,", ",")
            line = line.replace(" !", "!")
            line = re.sub("[.]+", ".", line)
            line = re.sub("[,]+", ",", line)
            line = re.sub("[!]+", "!", line)
            if len(line.strip()) > 0:
                lines.append(line)
    corpus = " ".join(lines)
    return corpus
	
def get_charmap(corpus):
    chars = list(set(corpus))
    chars.sort()
    charmap = {c: i for i, c in enumerate(chars)}
    return chars, charmap


def map_corpus(corpus, charmap):
    return np.array([charmap[c] for c in corpus], dtype=np.int64)


def to_text(line, charset):
    return "".join([charset[c] for c in line])