import numpy as np


def create_word_dict(corpus):
    vocab_dict = {}
    all_word_count = 0
    for sentence in corpus:
        for word in sentence:
            if(word not in vocab_dict):
                vocab_dict[word] = all_word_count
                all_word_count = all_word_count + 1
    return vocab_dict


def count_vocab(word_dict):
    return len(word_dict)
    

def create_co_matrix(corpus, windows_size, vocab_size, word_dict):
    co_matrix = np.zeros((vocab_size, vocab_size))
    for sentence_index in range(len(corpus)):
        for word_index in range(len(corpus[sentence_index])):
            left_most_word = word_index - windows_size
            right_most_word = word_index + windows_size
            for i in range(left_most_word, right_most_word + 1):
                if( i >=  0  and  i  !=  word_index  and  i < len (corpus[ sentence_index ])):
                    context_word = corpus[sentence_index][i]
                    center_word = corpus[sentence_index][word_index]
                    context_word_id = word_dict.get(context_word)
                    center_word_id = word_dict.get(center_word)
                    co_matrix[center_word_id][context_word_id] = co_matrix[center_word_id][context_word_id] + 1
    return co_matrix


corpus = np.array([["I", "go", "to", "school", "and", "I", "think"]])
word_dict = create_word_dict(corpus)
vocab_count = count_vocab(word_dict)
co_matrix = create_co_matrix(corpus, 1, vocab_count, word_dict)
print(word_dict)
print("Co-matrix before dimension reduction: ")
print(co_matrix)
print()
print("U matrix after dimension reduction: ")
U, S, V = np.linalg.svd(co_matrix)
print(U)





