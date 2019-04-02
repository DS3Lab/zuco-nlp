from pycorenlp import StanfordCoreNLP
import io
import re
import yaml
import data_helpers


def annotate_pos_tags(text):
    output = nlp.annotate(text, properties={'annotators': "pos", 'outputFormat': 'json', 'tokenize.whitespace': 'true'})
    return output


nlp = StanfordCoreNLP('http://localhost:9000')
# start corenlp: $ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

# subtask = "2.12classes"
task = "zuco_nr_cleanphrases"
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

datasets, x, x_text, y, vocab_processor, _, _ = data_helpers.get_processed_dataset(cfg, task)
sentences_with_tags = [a for a in vocab_processor.reverse(x)]

# entities_by_file = data_helpers.load_entities_by_file(cfg["datasets"]["subtask" + subtask]["entities_file_path"])
# reverse_labels = [entities_by_file[x][2] for x in [re.findall('/([0-9]+).txt$', filename)[0] for filename in datasets['filenames']]]

pos_tags_all = []

# for sentence, reverse_label in zip(sentences_with_tags, reverse_labels):
for sentence in sentences_with_tags:
    # print("Original sentence: {}".format(sentence))
    sentence = sentence.split(" ")
    tagged_by_parser = [w not in ("e", "<UNK>") for w in sentence]
    clean_sentence_words = [sentence[i] for i in range(0, len(sentence)) if tagged_by_parser[i]]
    # Reverse sentence again if sentence was reversed
    # clean_sentence = " ".join(clean_sentence_words[::-1]) if reverse_label=="REVERSE" else " ".join(clean_sentence_words)
    clean_sentence = " ".join(clean_sentence_words)
    # print("Sentence in correct order: {}".format(clean_sentence))
    annotation_json = annotate_pos_tags(clean_sentence)
    annotated_pos_tags = [x['pos'] for x in annotation_json['sentences'][0]['tokens']]

    # if reverse_label == "REVERSE":
    #	print("REVERSED")
    #	# Reverse annotations to keep consistency
    #	annotated_pos_tags = annotated_pos_tags[::-1]

    # print("Annotated POS tags: {}".format(annotated_pos_tags))

    if (len(annotated_pos_tags) != len(clean_sentence_words)):
        print("LENGTHS DIFFER. Sentence: {} vs. Tags: {}".format(len(clean_sentence_words), len(annotated_pos_tags)))
        print(sentence)
        break
    pos_tags = [None] * len(sentence)
    j = 0
    for i in range(len(sentence)):
        if tagged_by_parser[i]:
            pos_tags[i] = annotated_pos_tags[j]
            j += 1
        else:
            if (sentence[i] == "e"):
                pos_tags[i] = "EEE"
            elif (sentence[i] == "<UNK>"):
                pos_tags[i] = "UNK"

    # print("Final POS tags: {}".format(pos_tags))
    pos_tags_all.append(pos_tags)

# OUTPUT ANNOTATIONS TO A FILE
output_file = io.open("./preprocessing/postags" + task + ".txt", mode="w", encoding="utf-8")
for annotation in pos_tags_all:
    print(" ".join(annotation), file=output_file)

output_file.close()
