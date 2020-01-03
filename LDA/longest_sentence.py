from BoW import get_articles, rootdir
text = '\n'.join(get_articles(rootdir))

# find the number of words of the longest sentence.
n = 0
previous_sentence_length = 0
current_sentence_length = 0
previous_sentence = ''
current_sentence = ''

for i in range(len(text)):
    current_sentence += text[i]
    if text[i] == ' ':
        n += 1
    elif text[i] == '.' \
            or text[i] == '!' \
            or text[i] == '?'\
            or text[i] == '\n':
        current_sentence_length = n+1
        if current_sentence_length > previous_sentence_length:
            previous_sentence_length = current_sentence_length
            previous_sentence = current_sentence
        n = 0
        current_sentence = ''

print('Longest sentence found is:')
print('>>', previous_sentence)
print('with a length of %d words.' % previous_sentence_length)

