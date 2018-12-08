
import sys;

import tensorflow as tf
import numpy;

# configure here
TEST_TRUMP = False
TRAIN_TRUMP = False
TEST_NO_TRUMP = True
TRAIN_NO_TRUMP = True
BATCHES = 4
PARTITION = 0.50
SET_SIZE = 600000
EXPERIMENT = "full_no_trump_estimation_tests"
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
#data = numpy.load(path + "train_data.npy");
#labels = numpy.load(path + "train_output.npy");
#test_data = numpy.load(path + "test_data.npy");
#test_labels = numpy.load(path + "test_output.npy");
#net_outputs = numpy.load(path + "finetuningtest_outputs.npy");

#labels = list(map(lambda x: x.argmax(), labels));
#test_labels = list(map(lambda x: x.argmax(), test_labels));
#net_outputs = list(map(lambda x: x.argmax(), net_outputs[0]));

(data_l, data_r, outputs, diffs, test_data_l, test_data_r, test_outputs, test_diffs) = dp.read_file("./../data/library", SET_SIZE, True, TRAIN_NO_TRUMP, TRAIN_TRUMP, TEST_NO_TRUMP, TEST_TRUMP, PARTITION);

numpy.save(path + "data_l", data_l);
numpy.save(path + "data_r", data_r);
numpy.save(path + "outputs", outputs);
numpy.save(path + "diffs", diffs);
numpy.save(path + "test_data_l", test_data_l);
numpy.save(path + "test_data_r", test_data_r);
numpy.save(path + "test_outputs", test_outputs);
numpy.save(path + "test_diffs", test_diffs);


#limit claffication test samples
#test_labels

# get sample counts
dp.save_distribution(path, dp.get_distribution(diffs), dp.get_distribution(test_diffs))

# calculate test set length
l = len(data);
batch_count = BATCHES;
data_batches = [];
data_batches_l = [];
data_batches_r = []
outputs_batches = [];

# separate data into batches
batch_size = int(l / batch_count);
for i in range(0, batch_count-1):
    data_batches_l.append(data_l[i * batch_size : (i + 1) * batch_size]);
    data_batches_r.append(data_r[i * batch_size : (i + 1) * batch_size]);
    outputs_batches.append(outputs[i * batch_size : (i + 1) * batch_size]);

data_batches.append(data[(batch_count - 1) * batch_size : l]);
outputs_batches.append(outputs[(batch_count - 1) * batch_size : l]);
print(len(data_batches[0]))

optimizer = tf.train.RMSPropOptimizer 
print(len(data));
print(len(test_data))

# create autoencoder
a = models.Autoencoder.build(208, [104, 52, 13], models.Model.cross_entropy_loss);


# pretrain each layer
a.pretrain(0.001, 0, 1000, data_batches, 0, 0.1, path + "{0}" , optimizer, 0.2, 15);

# create classifier
c = models.Classifier(a, 2);
# train whole network
c.train(data_batches_l, data_batches_r, outputs_batches, 0.0001, 15000, 0.0001, path +"/finetuning", data_l, data_r, outputs, diffs, test_data_l, test_data_r, test_output, test_diffs, dp.suit_count_for_params(TRAIN_NO_TRUMP, TRAIN_TRUMP), dp.suit_count_for_params(TEST_NO_TRUMP, TEST_TRUMP), models.Model.mse_loss, 25, experiment_name);

# evaluate results
# print(c.test(data, outputs));
# print(c.test(test_data, test_outputs));
# print(c.suit_based_accurancy(data, outputs, dp.suit_count_for_params(TRAIN_NO_TRUMP, TRAIN_TRUMP)));
# print(c.suit_based_accurancy(test_data, test_outputs, dp.suit_count_for_params(TEST_NO_TRUMP, TEST_TRUMP)));
c.save_model(experiment_name);


