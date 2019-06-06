#!/usr/bin/env bash

ORIGINAL_DATA_DIRECTORY="../data/original"
PROCESSED_DATA_DIRECTORY="../data/processed/txt"
CSV_DATA_DIRECTORY="../data/processed/csv"

TRAIN_DATA="$ORIGINAL_DATA_DIRECTORY/twitter-training-data.txt"
PROC_TRAIN_DATA="$PROCESSED_DATA_DIRECTORY/proc-twitter-training-data.txt"
PROC_CSV_TRAIN="$CSV_DATA_DIRECTORY/proc-twitter-training-data.csv"

DEV_DATA="$ORIGINAL_DATA_DIRECTORY/twitter-dev-data.txt"
PROC_DEV_DATA="$PROCESSED_DATA_DIRECTORY/proc-twitter-dev-data.txt"
PROC_CSV_DEV="$CSV_DATA_DIRECTORY/proc-twitter-dev-data.csv"

TEST_DATA1="$ORIGINAL_DATA_DIRECTORY/twitter-test1.txt"
PROC_TEST_DATA1="$PROCESSED_DATA_DIRECTORY/proc-twitter-test1.txt"
PROC_CSV_TEST1="$CSV_DATA_DIRECTORY/proc-twitter-test1.csv"

TEST_DATA2="$ORIGINAL_DATA_DIRECTORY/twitter-test2.txt"
PROC_TEST_DATA2="$PROCESSED_DATA_DIRECTORY/proc-twitter-test2.txt"
PROC_CSV_TEST2="$CSV_DATA_DIRECTORY/proc-twitter-test2.csv"

TEST_DATA3="$ORIGINAL_DATA_DIRECTORY/twitter-test3.txt"
PROC_TEST_DATA3="$PROCESSED_DATA_DIRECTORY/proc-twitter-test3.txt"
PROC_CSV_TEST3="$CSV_DATA_DIRECTORY/proc-twitter-test3.csv"

echo "Started processing the input tweets.. "

if [ ! -f "$PROC_TRAIN_DATA" ]; then
	echo "Ups! preproc_train_file not found. We'll create one now.."
	touch "$PROC_TRAIN_DATA"
fi

if [ ! -f "$PROC_CSV_TRAIN" ]; then
	echo "Ups! preproc_csv_train_file not found. We'll create one now.."
	touch "$PROC_CSV_TRAIN"
fi

if [ ! -f "$PROC_DEV_DATA" ]; then
	echo "Ups! preproc_dev_data not found. We'll create one now.."
	touch "$PROC_DEV_DATA"
fi

if [ ! -f "$PROC_CSV_DEV" ]; then
	echo "Ups! preproc_csv_dev_file not found. We'll create one now.."
	touch "$PROC_CSV_DEV"
fi

if [ ! -f "$PROC_TEST_DATA1" ]; then
	echo "Ups! preproc_test_data1 not found. We'll create one now.."
	touch "$PROC_TEST_DATA1"
fi

if [ ! -f "$PROC_CSV_TEST1" ]; then
	echo "Ups! preproc_csv_test_data1 not found. We'll create one now.."
	touch "$PROC_CSV_TEST1"
fi

if [ ! -f "$PROC_TEST_DATA2" ]; then
	echo "Ups! preproc_test_data2 not found. We'll create one now.."
	touch "$PROC_TEST_DATA2"
fi

if [ ! -f "$PROC_CSV_TEST2" ]; then
	echo "Ups! preproc_csv_test_data2 not found. We'll create one now.."
	touch "$PROC_CSV_TEST2"
fi

if [ ! -f "$PROC_TEST_DATA3" ]; then
	echo "Ups! preproc_test_data3 not found. We'll create one now.."
	touch "$PROC_TEST_DATA3"
fi

if [ ! -f "$PROC_CSV_TEST3" ]; then
	echo "Ups! preproc_csv_test_data3 not found. We'll create one now.."
	touch "$PROC_CSV_TEST3"
fi

echo ""
echo "Filling in the new files with processed data..."

echo ""
echo "Processing the training data.."
./process_dataV3.py --orig_file "$TRAIN_DATA" --preproc_file "$PROC_TRAIN_DATA" 
./txt_to_csv.py --txt_file "$PROC_TRAIN_DATA" --csv_file "$PROC_CSV_TRAIN" --corpus_type "Tweet" --target "Sentiment" --numeric_target --target_types "negative" "neutral" "positive"
echo "Finished processing training data!"
echo "For your inspection, the processed train data was written in ($PROC_TRAIN_DATA)."
echo "A CSV version of the file is in ($PROC_CSV_TRAIN)"

echo ""
echo "Processing the development data file..."
./process_dataV3.py --orig_file "$DEV_DATA" --preproc_file "$PROC_DEV_DATA" 
./txt_to_csv.py --txt_file "$PROC_DEV_DATA" --csv_file "$PROC_CSV_DEV" --corpus_type "Tweet" --target "Sentiment" --numeric_target --target_types "negative" "neutral" "positive"
echo "Finished processing development data!"
echo "For your inspection, the processed development data was written in ($PROC_DEV_DATA)."
echo "A CSV version of the file is in ($PROC_CSV_DEV)"

echo ""
echo "Processing the 1st test data file..."
./process_dataV3.py --orig_file "$TEST_DATA1" --preproc_file "$PROC_TEST_DATA1" 
./txt_to_csv.py --txt_file "$PROC_TEST_DATA1" --csv_file "$PROC_CSV_TEST1" --corpus_type "Tweet" --target "Sentiment" --numeric_target --target_types "negative" "neutral" "positive"
echo "Finished processing the 1st test data!"
echo "For your inspection, the processed 1st test data was written in ($PROC_TEST_DATA1)."
echo "A CSV version of the file is in ($PROC_CSV_TEST1)"

echo ""
echo "Processing the 2nd test data file..."
./process_dataV3.py --orig_file "$TEST_DATA2" --preproc_file "$PROC_TEST_DATA2" 
./txt_to_csv.py --txt_file "$PROC_TEST_DATA2" --csv_file "$PROC_CSV_TEST2" --corpus_type "Tweet" --target "Sentiment" --numeric_target --target_types "negative" "neutral" "positive"
echo "Finished processing the 2nd test data!"
echo "For your inspection, the processed 2nd test data was written in ($PROC_TEST_DATA2)."
echo "A CSV version of the file is in ($PROC_CSV_TEST2)"

echo ""
echo "Processing the 3rd test data file..."
./process_dataV3.py --orig_file "$TEST_DATA3" --preproc_file "$PROC_TEST_DATA3" 
./txt_to_csv.py --txt_file "$PROC_TEST_DATA3" --csv_file "$PROC_CSV_TEST3" --corpus_type "Tweet" --target "Sentiment" --numeric_target --target_types "negative" "neutral" "positive"
echo "Finished processing the 3rd test data!"
echo "For your inspection, the processed 3rd test data was written in ($PROC_TEST_DATA3)."
echo "A CSV version of the file is in ($PROC_CSV_TEST3)"
