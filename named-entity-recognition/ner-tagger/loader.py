import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            word[0] = zero_digits(word[0]) if zeros else word[0]
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def cognitive_features(s):
    """
    ***Eye-tracking features***
    column in input data    gaze feature
    2	Total fixation duration
    3	n fixations
    4	First fixation duration
    5	First pass duration
    6	Fixation probability
    7   n refixations
    8   Re-read probability
    9   Mean fixation duration
    10	Total regression-from duration
    11	w-2 fixation probability
    12	w-1 fixation probability
    13	w+1 fixation probability
    14	w+2 fixation probability
    15	w-2 fixation duration
    16	w-1 fixation duration
    17	w+1 fixation duration
    18	w+2 fixation duration
    """

    tfd = int(s[1])
    n_fix = int(s[2])
    ffd = int(s[3])
    fpd = int(s[4])
    fix_prob = int(s[5])
    n_ref = int(s[6])
    rrp = int(s[7])
    mfd = int(s[8])
    rfd = int(s[9])
    wm2_fix_prob = int(s[10])
    wm1_fix_prob = int(s[11])
    wp1_fix_prob = int(s[12])
    wp2_fix_prob = int(s[13])
    wm2_fix_dur = int(s[14])
    wm1_fix_dur = int(s[15])
    wp1_fix_dur = int(s[16])
    wp2_fix_dur = int(s[17])

    """
    ***EEG features***
    column in input data    eeg feature
    19	FFD t1 (First fixation duration (FFD) on frequency band tetha-1)
    20	FFD t2
    21	FFD a1
    22	FFD a2
    23	FFD b1
    24  FFD b2
    25  FFD g1
    26  FFD g2
    """

    ffd_t1 = int(s[18])
    ffd_t2 = int(s[19])
    ffd_a1 = int(s[20])
    ffd_a2 = int(s[21])
    ffd_b1 = int(s[22])
    ffd_b2 = int(s[23])
    ffd_g1 = int(s[24])
    ffd_g2 = int(s[25])

    return tfd, n_fix, ffd, fpd, fix_prob, n_ref, rrp, mfd, rfd, wm2_fix_prob, wm1_fix_prob, wp1_fix_prob, wp2_fix_prob, wm2_fix_dur, wm1_fix_dur, wp1_fix_dur, wp2_fix_dur, ffd_t1, ffd_t2, ffd_a1, ffd_a2, ffd_b1, ffd_b2, ffd_g1, ffd_g2


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    # todo: check if this works
    tfd = [cognitive_features(w)[0] for w in str_words]
    n_fix = [cognitive_features(w)[1] for w in str_words]
    ffd = [cognitive_features(w)[2] for w in str_words]
    fpd = [cognitive_features(w)[3] for w in str_words]
    fix_prob = [cognitive_features(w)[4] for w in str_words]
    n_ref = [cognitive_features(w)[5] for w in str_words]
    rrp = [cognitive_features(w)[6] for w in str_words]
    mfd = [cognitive_features(w)[7] for w in str_words]
    rfd = [cognitive_features(w)[8] for w in str_words]
    wm2_fix_prob = [cognitive_features(w)[9] for w in str_words]
    wm1_fix_prob = [cognitive_features(w)[10] for w in str_words]
    wp1_fix_prob = [cognitive_features(w)[11] for w in str_words]
    wp2_fix_prob = [cognitive_features(w)[12] for w in str_words]
    wm2_fix_dur = [cognitive_features(w)[13] for w in str_words]
    wm1_fix_dur = [cognitive_features(w)[14] for w in str_words]
    wp1_fix_dur = [cognitive_features(w)[15] for w in str_words]
    wp2_fix_dur = [cognitive_features(w)[16] for w in str_words]
    ffd_t1 = [cognitive_features(w)[17] for w in str_words]
    ffd_t2 = [cognitive_features(w)[18] for w in str_words]
    ffd_a1 = [cognitive_features(w)[19] for w in str_words]
    ffd_a2 = [cognitive_features(w)[20] for w in str_words]
    ffd_b1 = [cognitive_features(w)[21] for w in str_words]
    ffd_b2 = [cognitive_features(w)[22] for w in str_words]
    ffd_g1 = [cognitive_features(w)[23] for w in str_words]
    ffd_g2 = [cognitive_features(w)[24] for w in str_words]

    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'tfd' : tfd,
        'n_fix' : n_fix,
        'ffd' : ffd,
        'fpd' : fpd,
        'fix_prob': fix_prob,
        'n_ref': n_ref,
        'rrp' : rrp,
        'mfd' : mfd,
        'rfd' : rfd,
        'wm2_fix_prob': wm2_fix_prob,
        'wm1_fix_prob': wm1_fix_prob,
        'wp1_fix_prob': wp1_fix_prob,
        'wp2_fix_prob': wp2_fix_prob,
        'wm2_fix_dur': wm2_fix_dur,
        'wm1_fix_dur': wm1_fix_dur,
        'wp1_fix_dur': wp1_fix_dur,
        'wp2_fix_dur': wp2_fix_dur,
        'ffd_t1': ffd_t1,
        'ffd_t2': ffd_t2,
        'ffd_a1': ffd_a1,
        'ffd_a2': ffd_a2,
        'ffd_b1': ffd_b1,
        'ffd_b2': ffd_b2,
        'ffd_g1': ffd_g1,
        'ffd_g2': ffd_g2
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tfd = [cognitive_features(w)[0] for w in s]
        n_fix = [cognitive_features(w)[1] for w in s]
        ffd = [cognitive_features(w)[2] for w in s]
        fpd = [cognitive_features(w)[3] for w in s]
        fix_prob = [cognitive_features(w)[4] for w in s]
        n_ref = [cognitive_features(w)[5] for w in s]
        rrp = [cognitive_features(w)[6] for w in s]
        mfd = [cognitive_features(w)[7] for w in s]
        rfd = [cognitive_features(w)[8] for w in s]
        wm2_fix_prob = [cognitive_features(w)[9] for w in s]
        wm1_fix_prob = [cognitive_features(w)[10] for w in s]
        wp1_fix_prob = [cognitive_features(w)[11] for w in s]
        wp2_fix_prob = [cognitive_features(w)[12] for w in s]
        wm2_fix_dur = [cognitive_features(w)[13] for w in s]
        wm1_fix_dur = [cognitive_features(w)[14] for w in s]
        wp1_fix_dur = [cognitive_features(w)[15] for w in s]
        wp2_fix_dur = [cognitive_features(w)[16] for w in s]
        ffd_t1 = [cognitive_features(w)[17] for w in s]
        ffd_t2 = [cognitive_features(w)[18] for w in s]
        ffd_a1 = [cognitive_features(w)[19] for w in s]
        ffd_a2 = [cognitive_features(w)[20] for w in s]
        ffd_b1 = [cognitive_features(w)[21] for w in s]
        ffd_b2 = [cognitive_features(w)[22] for w in s]
        ffd_g1 = [cognitive_features(w)[23] for w in s]
        ffd_g2 = [cognitive_features(w)[24] for w in s]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tfd' : tfd,
            'n_fix' : n_fix,
            'ffd' : ffd,
            'fpd' : fpd,
            'fix_prob': fix_prob,
            'n_ref': n_ref,
            'rrp': rrp,
            'mfd': mfd,
            'rfd': rfd,
            'wm2_fix_prob': wm2_fix_prob,
            'wm1_fix_prob': wm1_fix_prob,
            'wp1_fix_prob': wp1_fix_prob,
            'wp2_fix_prob': wp2_fix_prob,
            'wm2_fix_dur': wm2_fix_dur,
            'wm1_fix_dur': wm1_fix_dur,
            'wp1_fix_dur': wp1_fix_dur,
            'wp2_fix_dur': wp2_fix_dur,
            'ffd_t1': ffd_t1,
            'ffd_t2': ffd_t2,
            'ffd_a1': ffd_a1,
            'ffd_a2': ffd_a2,
            'ffd_b1': ffd_b1,
            'ffd_b2': ffd_b2,
            'ffd_g1': ffd_g1,
            'ffd_g2': ffd_g2,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
