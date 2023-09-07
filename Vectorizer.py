import string


class Vectorizer:

    # Padroniza texto, removendo pontuações e colocando todos os caracteres em caixa baixa
    def __init__(self):
        self.vocabulary = None
        self.inverse_vocabulary = None

    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    # Cria tokens a partir de um texto
    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    # Cria o vocabulário do texto
    def make_vocabulary(self, dataset):
        self.vocabulary = {"": 0, "[UNK]": 1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict((value, key) for key, value in self.vocabulary.items())

    # Codifica conforme definição do vocabulário
    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    # Decodifica conforme definição do vocabulário
    def decode(self, coded_sequence):
        return " ".join(self.inverse_vocabulary.get(index, "[UNK]") for index in coded_sequence)
