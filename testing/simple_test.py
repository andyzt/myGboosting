from sklearn.metrics import roc_auc_score
import logging
import time
import sys


def split_data_to_train_and_test(set_dir, subsample_n, test_part, has_header=False, seed=42):
    raise NotImplemented()


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

    parser.add_argument('-w', '--work-dir', default="testing_dir_" + time.strftime("%Y-%m-%d_%H-%M-%S"),
                        help='Directory to store work files. Files in directory may be overwritten')

    context = parser.parse_args(sys.argv[1:])

    if context.set is not None:
        train_file, test_file = split_data_to_train_and_test(context.set,
                                                             context.subsample,
                                                             context.test_part,
                                                             context.has_header)
    else:
        assert context.train_set is not None and context.test_set is not None, \
            "Please specify set or both train and test sets"
        train_file = context.train_set
        test_file = context.test_set



