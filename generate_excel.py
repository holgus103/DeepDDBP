
import sys;

import tensorflow as tf
import numpy;
# configure here
TEST_TRUMP = True
TRAIN_TRUMP = True
TEST_NO_TRUMP = False
TRAIN_NO_TRUMP = False
BATCHES = 4
PARTITION = 0.50
SET_SIZE = 200000
EXPERIMENT = "trump_l_104_52_13_p104_c_2_no_draws"
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
a = models.Autoencoder.build(208, [104, 52, 13], models.Model.cross_entropy_loss);


# create classifier
c = models.Classifier(a, 2);


c.restore_model("no_trump_l_104_52_13_p104_c_2_no_draws at 21100");
#success = False;
#cnt = 0;
#while(not success):
comparables = dp.labeled_dictionary(data, labels, 7);
samples = dp.labeled_dictionary(test_data, test_labels, 10);
l = []
c.classify_sequential([samples[i][j] for i in range(0,14) for j in range(0, 10)], comparables, [ i for i in range(0, 14) for x in range(0, 10)], [ i for i in range(0, 14) for x in range(0, 10)], 0.05, models.Autoencoder.UP, True, l)
# generate actual data
res = [[[0 if abs(test[0] - test[1]) < 0 else -1 if test[1] > test[0] else 1 for test in value] for value in sample ] for sample in l ]

file = open('res_no_draws.csv','w') 
for sample in range(0,140):
    file.write("sample, {0} \n".format(int(sample/10)))
    for value in range(0, len(res[sample])):
        file.write("comparing with {0}, ".format(value))
        for test in range(0, len(res[sample][value])):
            file.write(str(res[sample][value][test]))
            if(test < len(res[sample][value])- 1):
                file.write(",");
        file.write("\n")

file.close()