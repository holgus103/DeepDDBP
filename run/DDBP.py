
import sys;

import tensorflow as tf

# configure here
TEST_TRUMP = False
TRAIN_TRUMP = False
TEST_NO_TRUMP = True
TRAIN_NO_TRUMP = True
BATCHES = 4
PARTITION = 0.5
SET_SIZE = 200
EXPERIMENT = "no_trump_hand_rotations_14out_100k_proper"



# main experiment code
sys.path.append("./../");

import models;
import data_parser as dp;

experiment_name = EXPERIMENT;
path = "./summaries/{0}/".format(experiment_name);

dp.initialize_random(experiment_name);

# import data
(data, test_data) = dp.read_file("./../data/sol100000.txt", SET_SIZE, True, TRAIN_NO_TRUMP, TRAIN_TRUMP, TEST_NO_TRUMP, TEST_TRUMP, PARTITION);

# d_train = dp.get_distribution(data, outputs);
# d_test = dp.get_distribution(test_data, test_outputs);
# dp.save_distribution(path, d_train, d_test);
optimizer = tf.train.RMSPropOptimizer 
print(len(data));
print(len(test_data))

# create autoencoder
a = models.Autoencoder.build(208, [52], models.Model.cross_entropy_loss);


# pretrain each layer
a.pretrain(0.001, 0, 1000, data, 0, 1, path + "{0}" , optimizer, 0.2, 15);

# create classifier
c = models.Classifier(a, 2);
# train whole network
c.train(data, test_data, 0.0001, 15000, 0.1, path +"/finetuning", data, outputs, test_data, test_outputs, dp.suit_count_for_params(TRAIN_NO_TRUMP, TRAIN_TRUMP), dp.suit_count_for_params(TEST_NO_TRUMP, TEST_TRUMP), models.Model.mse_loss, 25, experiment_name);

# evaluate results
# print(c.test(data, outputs));
# print(c.test(test_data, test_outputs));
# print(c.suit_based_accurancy(data, outputs, dp.suit_count_for_params(TRAIN_NO_TRUMP, TRAIN_TRUMP)));
# print(c.suit_based_accurancy(test_data, test_outputs, dp.suit_count_for_params(TEST_NO_TRUMP, TEST_TRUMP)));
c.save_model(experiment_name);


