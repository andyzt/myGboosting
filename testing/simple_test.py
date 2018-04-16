from sklearn.metrics import roc_auc_score
import logging
import time
import sys
import os
import os.path
import random


def split_data_to_train_and_test(set_dir, work_dir, subsample_n, test_part, has_header=False, seed=42):
    f = open(os.path.join(set_dir, "data.csv"))
    header = None
    if has_header:
        header = f.readline()[:-1]
    lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines if x.strip() != ""]
    random.seed(seed)
    lines = random.shuffle(lines)
    if subsample_n is not None:
        lines = lines[:subsample_n]
    n = len(lines)
    test_n = int(n * test_part)
    assert test_n > 0, "Test set is empty. Please increase test part or subsample"
    train_filename = os.path.join(work_dir, "train.csv")
    with open(train_filename, "w") as f:
        if header is not None:
            f.write(header + "\n")
        f.writelines((x + "\n" for x in lines[test_n:]))

    test_filename = os.path.join(work_dir, "test.csv")
    with open(test_filename, "w") as f:
        if header is not None:
            f.write(header + "\n")
        f.writelines((",".join(x.split(",")[:-1]) + "\n" for x in lines[:test_n]))

    test_labels_filename = os.path.join(work_dir, "test_labels.csv")
    with open(test_filename, "w") as f:
        if header is not None:
            f.write(header + "\n")
        f.writelines((x.split(",")[:-1] + "\n" for x in lines[:test_n]))

    return train_filename, test_filename, test_labels_filename


if __name__ == "__main__":
    root_logger = logging.getLogger('')
    file_logger = logging.StreamHandler()
    file_logger.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d %(message)s'))
    root_logger.addHandler(file_logger)
    root_logger.setLevel(logging.DEBUG)

    import argparse
    parser = argparse.ArgumentParser(description='Trains model on given dataset then evaluates on it')
    parser.add_argument('-s', '--set', help='Dir with data.csv and columns.csv (optional)', default=None)
    parser.add_argument('-h', '--has-header', help='Specifie if data files has header line', action='store_const')
    parser.add_argument('--target', help='Target column name. Unused if no column file specified. '
                                         'Default: target', default="target")

    parser.add_argument('-n', '--subsample', help='Number of rows to be used.', default=None, type=int)
    parser.add_argument('-p', '--test-part', help='% of test set. Default: 50', default=50.0, type=float)
    parser.add_argument('--seed', help='Random seed for splitting and sampling', default=42, type=int)
    parser.add_argument('--train-args', help='Additional args to be passed while training', default="")
    parser.add_argument('--test-args', help='Additional args to be passed while testing', default="")

    parser.add_argument('--train-set', help='Overwrites set option', default=None)
    parser.add_argument('--test-set', help='Overwrites set option', default=None)
    parser.add_argument('--test-set-labels', help='Overwrites set option', default=None)

    parser.add_argument('-w', '--work-dir', default="testing_dir_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        help='Directory to store work files. Files in directory may be overwritten')

    context = parser.parse_args(sys.argv[1:])

    if context.set is not None:
        train_file, test_file, test_labels_file = split_data_to_train_and_test(context.set,
                                                                               context.work_dir,
                                                                               context.subsample,
                                                                               context.test_part,
                                                                               context.has_header)
    else:
        assert context.train_set is not None and context.test_set is not None and context.test_set_labels is not None, \
            "Please specify set or both train and test sets"
        train_file = context.train_set
        test_file = context.test_set
        test_labels_file = context.test_set_labels
    print(train_file, test_file, test_labels_file)