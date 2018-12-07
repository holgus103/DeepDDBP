
EXPERIMENT = "no_trump_rotations_margin"
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

#data = numpy.load(path + "train_data.npy");
#labels = numpy.load(path + "train_output.npy");
test_data = numpy.load(path + "test_data.npy");
test_labels = numpy.load(path + "test_output.npy");


test_labels = list(map(lambda x: x.argmax(), test_labels));

(test_samples_l, test_samples_r, test_outsputs, test_diffs) = dp.generate_random_pairs(test_data, test_labels, 400000);


# create autoencoder
a = models.Autoencoder.build(208, [104, 52, 13], models.Model.cross_entropy_loss);


# create classifier
c = models.Classifier(a, 2);

#    def test(self, data_l, data_r, desired_output, diffs, margin = 0.4):
    
    


