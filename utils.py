import keras.backend as K
import numpy as np


def read_sentences(path):
    sentences = []
    sentence = []

    for line in open(path, 'r'):
        line = line.strip()

        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            sentence.append([word[0], word[3]])

    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    return sentences


def convert_iob1_to_iob2(tags):
    new_tags = []

    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            prefix = tag.split('-')[0]
            if len(tag.split('-')) != 2 or prefix not in ['I', 'B']:
                raise Exception('Invalid IOB format!')
            if prefix == 'B':
                new_tags.append(tag)
            elif i == 0 or tags[i - 1] == 'O':
                new_tags.append('B' + tag[1:])
            elif tags[i - 1][1:] == tag[1:]:
                new_tags.append(tag)
            else:
                new_tags.append('B' + tag[1:])

    return new_tags


def convert_iob2_to_iobes(tags):
    new_tags = []

    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tags[i].split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tags[i].split('-')[0] == 'I':
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB2 format!')

    return new_tags


def convert_tags(sentences):
    for sentence in sentences:
        tags = [line[1] for line in sentence]
        tags = convert_iob1_to_iob2(tags)
        tags = convert_iob2_to_iobes(tags)
        for line, tag in zip(sentence, tags):
            line[1] = tag


def add_auxiliary_information(sentences):
    count = 0

    for i in range(len(sentences)):
        length = len(sentences[i])
        sentences[i] = {
            'sentence': sentences[i],
            'start': count,
            'stop': count + length
        }
        count += length


def create_tag_mapping(datasets):
    tag_set = set()

    for dataset in datasets:
        for sentences in dataset:
            for line in sentences:
                tag_set.add(line[1])

    tag_idx = {}
    idx_tag = {}

    for tag in tag_set:
        tag_idx[tag] = len(tag_idx)
        idx_tag[len(idx_tag)] = tag

    tag_idx['<PAD>'] = len(tag_idx)
    idx_tag[len(idx_tag)] = '<PAD>'

    return tag_idx, idx_tag


def create_word_mapping(path):
    word_embeddings = []
    word_idx = {}
    idx_word = {}

    for line in open(path, 'r'):
        split = line.strip().split()

        word_idx[split[0]] = len(word_idx)
        idx_word[len(idx_word)] = split[0]
        word_embeddings.append(np.asarray(split[1:], dtype='float32'))

    embedding_size = len(word_embeddings[-1])

    word_idx['<PAD>'] = len(word_idx)
    idx_word[len(idx_word)] = '<PAD>'
    word_embeddings.append(np.zeros(embedding_size))

    word_idx['<UNK>'] = len(word_idx)
    idx_word[len(idx_word)] = '<UNK>'
    word_embeddings.append(np.random.uniform(-0.25, 0.25, embedding_size))

    word_embeddings = np.asarray(word_embeddings)

    return word_idx, idx_word, word_embeddings


def create_char_mapping(datasets):
    char_set = set()

    for dataset in datasets:
        for sentences in dataset:
            for line in sentences:
                for char in line[0]:
                    char_set.add(char)

    char_idx = {}
    idx_char = {}

    for char in char_set:
        char_idx[char] = len(char_idx)
        idx_char[len(idx_char)] = char

    char_idx['<PAD>'] = len(char_idx)
    idx_char[len(idx_char)] = '<PAD>'

    return char_idx, idx_char


def create_case_mapping():
    case_idx = {
        'entirely_digit': 0,
        'mainly_digit': 1,
        'contains_digit': 2,
        'all_lower': 3,
        'all_upper': 4,
        'initial_upper': 5,
        'other': 6,
        '<PAD>': 7
    }

    return case_idx, np.identity(len(case_idx), dtype='float32')


def get_casing(word):
    case_idx, _ = create_case_mapping()

    num_digit = 0
    digit_rate = 0
    if word != '<PAD>':
        for char in word:
            if char.isdigit():
                num_digit += 1
        digit_rate = float(num_digit / len(word))

    if word == '<PAD>':
        case = '<PAD>'
    elif word.isdigit():
        case = 'entirely_digit'
    elif digit_rate > 0.5:
        case = 'mainly_digit'
    elif num_digit > 0:
        case = 'contains_digit'
    elif word.islower():
        case = 'all_lower'
    elif word.isupper():
        case = 'all_upper'
    elif word[0].isupper():
        case = 'initial_upper'
    else:
        case = 'other'

    return case_idx[case]


def get_max_word_length(datasets):
    max_word_length = 0

    for sentences in datasets:
        for sentence in sentences:
            for word in sentence:
                max_word_length = max(max_word_length, len(word[0]))

    return max_word_length


def create_batch_data(batch, max_word_length, word_idx, char_idx, tag_idx, features, gazetteers):
    data = {
        'word': [],
        'char': [],
        'case': [],
        'lengths': [],
        'tag': [],
        'gaze': [],  # One-hot gaze tags
        'gazetteers': [],  # True gaze tags
        'features': [],  # One-hot features
        'shape': [],  # True shape features
        'position': []  # True position features
    }

    max_sentence_length = len(batch[0]['sentence'])

    for i, element in enumerate(batch):
        data['lengths'].append(len(element['sentence']))

        gaze = gazetteers[element['start']: element['stop']]
        feature = features[element['start']: element['stop']]

        shape = feature[:, 45:196]
        position = feature[:, 196:]

        gaze = np.hstack((gaze, np.zeros((len(gaze), 1))))
        shape = np.hstack((shape, np.zeros((len(shape), 1))))
        position = np.hstack((position, np.zeros((len(position), 1))))

        if len(element['sentence']) < max_sentence_length:
            gaze = np.vstack((gaze, np.zeros((max_sentence_length - len(element['sentence']), len(gaze[0])))))
            shape = np.vstack((shape, np.zeros((max_sentence_length - len(element['sentence']), len(shape[0])))))
            position = np.vstack((position, np.zeros((max_sentence_length - len(element['sentence']), len(position[0])))))

            for j in range(len(element['sentence']), max_sentence_length):
                gaze[j][-1] = 1
                shape[j][-1] = 1
                position[j][-1] = 1

            element['sentence'] += [['<PAD>', '<PAD>']] * (max_sentence_length - len(element['sentence']))

        batch[i]['sentence'] = [['<PAD>', '<PAD>']] + element['sentence'] + [['<PAD>', '<PAD>']]
        gaze = np.vstack((np.zeros(len(gaze[0])), gaze, np.zeros(len(gaze[0]))))
        shape = np.vstack((np.zeros(len(shape[0])), shape, np.zeros(len(shape[0]))))
        position = np.vstack((np.zeros(len(position[0])), position, np.zeros(len(position[0]))))

        gaze[0][-1] = 1
        gaze[-1][-1] = 1

        shape[0][-1] = 1
        shape[-1][-1] = 1

        position[0][-1] = 1
        position[-1][-1] = 1

        data['gaze'].append(gaze)
        data['features'].append(np.hstack((shape, position)))
        data['gazetteers'].append(np.argmax(gaze, axis=-1))
        data['shape'].append(np.argmax(shape, axis=-1))
        data['position'].append(np.argmax(position, axis=-1))

    for element in batch:
        data_word = []
        data_char = []
        data_case = []
        data_tag = []

        for word in element['sentence']:
            word_char = []

            if word_idx.get(word[0]) is not None:
                data_word.append(word_idx[word[0]])
            elif word_idx.get(word[0].lower()) is not None:
                data_word.append(word_idx[word[0].lower()])
            else:
                data_word.append(word_idx['<UNK>'])

            if word[0] == '<PAD>':
                word_char = [char_idx['<PAD>']] * max_word_length
            else:
                for char in word[0]:
                    word_char.append(char_idx[char])
                if len(word[0]) < max_word_length:
                    word_char += [char_idx['<PAD>']] * (max_word_length - len(word[0]))

            data_char.append(word_char)
            data_case.append(get_casing(word[0]))
            data_tag.append(tag_idx[word[1]])

        data['word'].append(data_word)
        data['char'].append(data_char)
        data['case'].append(data_case)
        data['tag'].append(data_tag)

    data['word'] = np.asarray(data['word'])
    data['char'] = np.asarray(data['char'])
    data['case'] = np.asarray(data['case'])
    data['lengths'] = np.asarray(data['lengths'])
    data['gaze'] = np.asarray(data['gaze'])
    data['features'] = np.asarray(data['features'])
    data['tag'] = np.asarray(np.expand_dims(data['tag'], -1))
    data['gazetteers'] = np.asarray(np.expand_dims(data['gazetteers'], -1))
    data['shape'] = np.asarray(np.expand_dims(data['shape'], -1))
    data['position'] = np.asarray(np.expand_dims(data['position'], -1))

    return data


def create_batches(sentences, batch_size, max_word_length, word_idx, char_idx, tag_idx, features, gazetteers):
    batches = []
    batch = []

    sentences.sort(key=lambda x: -len(x['sentence']))

    for sentence in sentences:
        if len(batch) == batch_size:
            data = create_batch_data(batch, max_word_length, word_idx, char_idx, tag_idx, features, gazetteers)
            batches.append(data)
            batch = []

        batch.append(sentence)

    if len(batch):
        data = create_batch_data(batch, max_word_length, word_idx, char_idx, tag_idx, features, gazetteers)
        batches.append(data)

    return batches


def weighted_sparse_categorical_crossentropy(target, output):
    weights = [3, 0.5, 3, 0]

    output /= K.sum(output, axis=-1, keepdims=True)
    output = K.clip(output, K.epsilon(), 1 - K.epsilon())

    loss = target * K.log(output) * weights
    loss = -K.sum(loss, -1)
    return loss
