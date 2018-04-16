#!/bin/bash
# please run "pip install kaggle" to get requirements
kaggle datasets download -d mlg-ulb/creditcardfraud -w
head -n 1 creditcard.csv   > column_names.csv
tail -n +2 creditcard.csv > data.csv
rm creditcard.csv

