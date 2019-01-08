# path = './CoNLL-2003/eng.train'
import re

def findAll(mystring, charac):
    charac_left = charac[0]
    charac_right = charac[1]
    matched = []
    left = []
    for i in range(len(mystring)):
        if mystring[i] == charac_left:
            left.append(i)
        if mystring[i] == charac_right:
            if len(left) == 1:
                matched.append(mystring[left[0]:i+1])
                left = []
            if len(left) > 1:
                matched.append(mystring[left[-1]:i+1])
                left = left[:-1]
    matched.sort(key=len, reverse=True)
    return matched

def removeBetweenBrackets(mystring):
    result1 = findAll(mystring, '()')
    result2 = findAll(mystring, '{}')
    result3 = findAll(mystring, '<>')
    result4 = findAll(mystring, '[]')
    for res in [result1, result2, result3, result4]:
        for x in res:
            mystring = mystring.replace(x, '')
    return mystring.strip()

def main(path, test=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    sentences = data.split('\n\n')
    TRAIN_DATA = []
    TEST_DATA = []
    entities = []
    for sent in sentences:
        sent = re.sub('cannot', 'can not', sent)
        tokens = sent.split('\n')
        sentence = []
        sentence_couple = []
        ent_sentence_spacy = []
        ents = []
        index = 0
        for x in tokens:
            x = re.sub('\'', '', x)
            x_split = x.split()
            if len(x) > 0 and len(x_split) >= 3:
                word = x_split[0]
                word = re.sub('U.S.', 'U.S', word)
                word = re.sub('ST.', 'ST', word)
                word = re.sub('-', '', word)
                word = re.sub('`', '', word)
                word = word.strip()
                sentence.append(word)
                try:
                    ent = x_split[-1]
                except IndexError:
                    print("Index Error: ", x_split)
                if ent != 'O':
                    ent_sentence_spacy.append((index, index+len(word), ent))
                    ent = ent.split('-')[1] # Remove B-... and I-... in front of the tags.
                if len(word) > 0:
                    ents.append(ent)
                    sentence_couple.append([word, ent])
                index += len(word)+1

            else:
                print('Short length x: ', x, ' . Removed.')

        # processed_sentence = " ".join(sentence)#.lower()
        TRAIN_DATA.append(sentence_couple)
        TEST_DATA.append(sentence)
        entities.append(ents)
    TRAIN_DATA = TRAIN_DATA[1:]
    if test:
        print("Done getting data !")
        print("There are %d testing sentences." % (len(sentences)))
        return TEST_DATA[1:], entities[1:]
    print("Done getting data !")
    print("There are %d training sentences." % (len(TRAIN_DATA)))
    return TRAIN_DATA