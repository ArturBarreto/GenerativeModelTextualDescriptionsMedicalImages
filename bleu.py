from nltk.translate.bleu_score import sentence_bleu

label = [
    "scan chest nodule".split()
]

predicted = "scan lung nodule".split()

print('Individual 1-gram: %f' % sentence_bleu(label, predicted, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(label, predicted, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(label, predicted, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(label, predicted, weights=(0, 0, 0, 1)))