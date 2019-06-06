#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, argparse, sys
import pandas as pd

def text_to_csv(TXT_FILE, CSV_FILE, CORPUS_TYPE, TARGET, NUMERIC_TARGET, TARGET_TYPES):
	for index, line in enumerate(TXT_FILE):
		line = line.split()
		if NUMERIC_TARGET == True:
			try:
				if line[1] == TARGET_TYPES[0]:
					line = [str(line[0]), 0, ' '.join(line[2:])]
				if line[1] == TARGET_TYPES[1]:
					line = [str(line[0]), 1, ' '.join(line[2:])]
				if line[1] == TARGET_TYPES[2]:
					line = [str(line[0]), 2, ' '.join(line[2:])]
			except:
				print ("Unexpected error occured:", sys.exc_info()[0])

		elif NUMERIC_TARGET == False:
			line = [str(line[0]), line[1], ' '.join(line[2:])]
		
		writer = csv.writer(CSV_FILE)
		if index == 0:
			writer.writerow((str("ID"), str(TARGET), str(CORPUS_TYPE)))
		writer.writerow(line)

def main(argv):
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-txt", "--txt_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-csv", "--csv_file", type=argparse.FileType('w'), default=sys.stdout)
	parser.add_argument("-ct", "--corpus_type", type=str, default=sys.stdin)
	parser.add_argument("-tg", "--target", type=str, default=sys.stdin)
	target_parser = parser.add_mutually_exclusive_group(required=False)
	target_parser.add_argument("-num_tg", "--numeric_target", action='store_true')
	target_parser.add_argument("-no_num_tg", "--no_numeric_target", action='store_false')
	parser.set_defaults(numeric_target=True)
	parser.add_argument("-tg_t", "--target_types", type=str, nargs='*')
	args = vars(parser.parse_args())
	text_to_csv(args["txt_file"], args["csv_file"], args["corpus_type"], args["target"], args["no_numeric_target"], args["target_types"])


if __name__ == "__main__":
    main(sys.argv)