import io
import yaml
import data_helpers

task = "zuco_nr_cleanphrases"
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

print("Loading dataset...")
datasets, x, x_text, y, vocab_processor, _, _ = data_helpers.get_processed_dataset(cfg, task)
sentences_with_tags = [a for a in vocab_processor.reverse(x)]
print("Dataset loaded...")

relative_positions_all = []

print("Calculating relative positions...")
for sentence in sentences_with_tags:
    # print("Original sentence: {}".format(sentence))
    sentence = sentence.split(" ")
    entity_tag_positions = [i for i, j in enumerate(sentence) if j == 'e']
    first_entity_pos = entity_tag_positions[1]
    second_entity_pos = entity_tag_positions[len(entity_tag_positions) - 2]
    last_tag_pos = entity_tag_positions[len(entity_tag_positions) - 1]

    relative_positions = [None] * len(sentence)
    for i in range(len(sentence)):
        if i > last_tag_pos:
            relative_positions[i] = (-999, -999)
            continue

        relative_first = max(i - first_entity_pos, 0)
        relative_second = min(i - second_entity_pos, 0)

        relative_positions[i] = (relative_first, relative_second)

    # print("Relative positions: {}".format(relative_positions))
    relative_positions_all.append(relative_positions)
print("Relative positions calculated")

# OUTPUT ANNOTATIONS TO A FILE
output_file_1 = io.open("./preprocessing/relative_positions_first" + task + ".txt", mode="w", encoding="utf-8")
output_file_2 = io.open("./preprocessing/relative_positions_second" + task + ".txt", mode="w", encoding="utf-8")
for relative_positions in relative_positions_all:
    print(" ".join([str(x[0]) for x in relative_positions]), file=output_file_1)
    print(" ".join([str(x[1]) for x in relative_positions]), file=output_file_2)

output_file_1.close()
output_file_2.close()
