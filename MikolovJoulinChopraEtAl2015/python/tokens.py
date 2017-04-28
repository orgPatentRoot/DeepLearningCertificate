# usecase_flg = 1 for predicting letters
#               2 for predicting words with fixed vocabulary size
#               3 for predicting words with cutoff for infrequent words

# Imports
import collections
import string

# Generate dictionary of tokens for text elements and convert text to tokens
def text_elements_to_tokens(usecase_flg, text_elements, vocabulary_size):
    dictionary = dict()
    if usecase_flg == 1:
        vocabulary_size = len(string.ascii_lowercase) + 2
        letters = ['UNK']
        letters += [' ']
        for letter in string.ascii_lowercase:
            letters += [letter]
        for letter in letters:
            dictionary[letter] = len(dictionary)
        vocab_size = vocabulary_size
    elif usecase_flg == 2:
        words = [['UNK', -1]]
        words.extend(collections.Counter(text_elements).most_common(vocabulary_size - 1))
        for word, _ in words:
            dictionary[word] = len(dictionary)
        vocab_size = vocabulary_size
    elif usecase_flg == 3:
        words = [['UNK', -1]]
        words.extend(collections.Counter(text_elements).most_common(len(text_elements)))
        frequencies = [i[1] for i in words]
        cutoff = word_frequency_cutoff
        while cutoff > 0:
            if cutoff in frequencies:
                idx = frequencies.index(cutoff)
                words = words[:idx]
                cutoff = 0
            else:
                cutoff -= 1
        for word, _ in words:
            dictionary[word] = len(dictionary)
        vocab_size = len(words)
    data = list()
    for text_element in text_elements:
        if text_element in dictionary:
            index = dictionary[text_element]
        else:
            index = dictionary['UNK']
        data.append(index)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, dictionary, reverse_dictionary, vocab_size

# Find text element for probability distribution over tokens
def token_to_text_element(probabilities, reverse_dictionary):
    return [reverse_dictionary[token] for token in np.argmax(probabilities, 1)]