
import sys;

import tensorflow as tf
import numpy;

# configure here
TEST_TRUMP = False
TRAIN_TRUMP = False
TEST_NO_TRUMP = True
TRAIN_NO_TRUMP = True
BATCHES = 4
PARTITION = 0.66
SET_SIZE = 600000
EXPERIMENT = "trump_altered_104enc_eta=0.004_no_draws"
# l - layers 208 - 104 - 52 - 13 x2
# p - pretrain 104
# c - classified 2x13 -> 2
# each - generate_random_pair_for_samples



# main experiment code
sys.path.append("./../");

import models;
import data_parser as dp;

experiment_name = EXPERIMENT;
path = "./summaries/{0}/".format(experiment_name);

dp.initialize_random(experiment_name);

# import data
data = numpy.load(path + "train_data.npy");
labels = numpy.load(path + "train_output.npy");
test_data = numpy.load(path + "test_data.npy");
test_labels = numpy.load(path + "test_output.npy");
net_outputs = numpy.load(path + "finetuningtest_outputs.npy");

labels = list(map(lambda x: x.argmax(), labels));
test_labels = list(map(lambda x: x.argmax(), test_labels));
net_outputs = list(map(lambda x: x.argmax(), net_outputs[0]));

#(data, labels, test_data, test_labels) = dp.read_file("./../data/library", SET_SIZE, True, TRAIN_NO_TRUMP, TRAIN_TRUMP, TEST_NO_TRUMP, TEST_TRUMP, PARTITION);
#(samples_l, samples_r, outputs, diffs) = dp.generate_random_pair_for_samples(data, labels)
#(test_samples_l, test_samples_r, test_outsputs, test_diffs) = dp.generate_random_pair_for_samples(test_data, test_labels);


#limit claffication test samples
#test_labels

# get sample counts
#dp.save_distribution(path, dp.get_distribution(diffs), dp.get_distribution(test_diffs))

# calculate test set length
#l = len(data);

# create autoencoder
a = models.Autoencoder.build(208, [[26, 26, 26, 26], [52], [13]], models.Model.cross_entropy_loss);
#a = models.Autoencoder.build(208, [104, 52, 13], models.Model.cross_entropy_loss);


# create classifier
c = models.Classifier(a, 2);


c.restore_model("trump_altered_104enc_eta=0.004_no_draws at 21000");
#success = False;
#cnt = 0;
#while(not success):


def get_stats(res):
    correct_correct, correct_wrong, wrong_correct, wrong_wrong = res;
    print("accuracy: {0}".format(float(correct_correct + wrong_correct) / sum(res)))
    print("confirmed correct: {0}".format(float(correct_correct) / (correct_correct + correct_wrong)))
    print("corrected wrongs: {0}".format(float(wrong_correct) / (wrong_correct + wrong_wrong)))

for val in [1000]:
    print("comparables size {0}".format(val))
    comparables = dp.labeled_dictionary(data, labels, val);
    print("sequential up")
    res = c.classify_sequential(test_data[0:10000], comparables, test_labels[0:10000], net_outputs[0:10000], 0.0 )
    get_stats(res);
    print("sequential down")
    res = c.classify_sequential(test_data[0:10000], comparables, test_labels[0:10000], net_outputs[0:10000], 0.0, models.Autoencoder.DOWN)
    get_stats(res);
    print("sequential dichotomy")
    res = c.classify_dichotomy_set(test_data[0:10000], net_outputs[0:10000], comparables, 0.0);
    print("accuracy: {0}".format(res))



