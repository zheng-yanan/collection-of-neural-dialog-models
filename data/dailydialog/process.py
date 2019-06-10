""" This file propcesses DailyDialog dataset.
	
	dailydialog.pkl constains a list dialog object.
	Each dialog object is a dictionary with keys {"topic", "utts"}.
	Each topic is an integer.
	Each utts is a list of utterance object.
	Each utterance object is a dictionary with keys {"floor", "text", "emot", "act"}.

	dailydialog_split.pkl is dictionary with keys {"train", "valid", "test"}.
	Each is a list of dialog objects, as above.
"""

import pickle as pkl
import sklearn


text_file 	= "dialogues_text.txt"
act_file 	= "dialogues_act.txt"
emot_file 	= "dialogues_emotion.txt"
topic_file 	= "dialogues_topic.txt"

text_f = open(text_file, "rb")
act_f = open(act_file, "rb")
emot_f = open(emot_file, "rb")
topic_f = open(topic_file, "rb")


lines = text_f.readlines()
acts = act_f.readlines()
emotions = emot_f.readlines()
topics = topic_f.readlines()


assert len(lines) == len(acts) == len(emotions) == len(topics)

print("Total Count: %d." % len(lines))

total_count = len(lines)
data = []

for idx in range(total_count):

	line = lines[idx].replace("\n", "").lower().strip().split("__eou__")[:-1]
	line = [line.strip() for line in line]
	# list of string

	act = acts[idx].replace(" \n", "").split(" ")
	act = [int(item)-1 for item in act]
	# list of int, [0, 3]

	emot = emotions[idx].replace(" \n", "").split(" ")
	emot = [int(item) for item in emot]
	# list of int, [0, 6]

	topic = topics[idx].replace("\n", "").split(" ")
	topic = int(topic[0])-1
	# int, [0, 9]

	if (len(line) == len(act) == len(emot)) == False:
		continue

	utt_num = len(line)
	dial_obj = {"topic": topic, "utts": []}

	flag = True
	for j in range(utt_num):

		floor = int(flag)
		flag = not flag

		utts_line = line[j].strip()
		utts_act = act[j]
		utts_emot = emot[j]

		utts_item = {"floor": floor, 
					"text": utts_line,
					"act": utts_act,
					"emot": utts_emot}

		dial_obj["utts"].append(utts_item)


	data.append(dial_obj)


print("Remaining Count: %d." % len(data))
new_data = sklearn.utils.shuffle(data)

f = open("dailydialog.pkl", "wb")
pkl.dump(new_data, f)


test = new_data[:1500]
valid = new_data[1500:3000]
train = new_data[3000:]

print("train/valid/test: %d/%d/%d" % (len(train), len(valid), len(test)))

ret = {"train": train, "valid": valid, "test": test}
	
ff = open("dailydialog_split.pkl", "wb")
pkl.dump(ret, ff)