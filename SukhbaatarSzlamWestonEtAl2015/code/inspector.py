import sys
import json
import os

filename = "../data/wiki/test/test-00000-of-00015.json"
no_lines = 10 if len(sys.argv)==1 else int(sys.argv[1]) 
with open(filename) as r:
	i = 1
	for line in r:
		line = json.loads(line)
		print (" ".join(line["string_sequence"]))
		print (" ".join(line["question_string_sequence"]))
		print (" ".join(line["raw_answers"]))
		print (line["answer_location"])
		print (line["full_match_answer_location"])
		if i > no_lines:
			break
		else:
			i = i+1
