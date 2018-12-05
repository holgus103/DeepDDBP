
EXPERIMENT = "no_trump_rotations_156enc_eta=0.002_no_draws"
import sys;
sys.path.append("./../");

import numpy
import models;
import data_parser as dp;
import point_count_methods as pm;

experiment_name = EXPERIMENT;
path = "./summaries/{0}/".format(experiment_name);

dp.initialize_random(experiment_name);

# import data
test_data = numpy.load(path + "test_data.npy");
test_labels = numpy.load(path + "test_output.npy");


test_labels = list(map(lambda x: x.argmax(), test_labels));

(test_samples_l, test_samples_r, test_outputs, test_diffs) = dp.generate_random_pairs(test_data, test_labels, 400000)
acc = 0
d = pm.wpc_dict();
d = pm.bamberger_dict()
d = pm.collet_dict()
d = pm.akq_points_dict()
for i in range(0, len(test_samples_l)):
    vl = pm.estimate_points(test_samples_l[i], d, [pm.assert_system]);
    vr = pm.estimate_points(test_samples_r[i], d, [pm.assert_system]);
    if ((test_outputs[i][0] > test_outputs[i][1]) and (vl > vr)) or ((test_outputs[i][0] < test_outputs[i][1]) and (vl < vr)):
        acc += 1;

print float(acc)/float(len(test_samples_l))
        
