import re
import numpy
import tensorflow as tf;
import random;
import _pickle as pickle
import os;
import pprint;


"""
    This module contains parsing operations for the datasets provided with this project.
    The datasets were originally published here: https://github.com/dds-bridge/dds/tree/develop/hands

    Input format described:
    TODO


"""

def parse(input, no_trump, trump):
    """
    Parse function used to process one line

    Parameters
    ----------
    input : string
        A deal encoded in the format specified above
    no_trump : bool
        Boolean value indicating whether No Trump games are to be skipped or not
    trump : bool
        Boolean value indicating whether Trump games are to be skipped or not

    Returns
    -------
    (list, list)
        Tuple of lists containing a set of generated inputs and a set of corresponding outputs
        """
    # mapping for chars in value string
    dict = {
        '0' : 0, 
        '1' : 1, 
        '2' : 2, 
        '3' : 3, 
        '4' : 4, 
        '5' : 5, 
        '6' : 6, 
        '7' : 7, 
        '8' : 8, 
        '9' : 9, 
        'A' : 10, 
        'B' : 11, 
        'C' : 12, 
        'D' : 13, 
    }
    t = input.split(":");
    data = [];
    outputs = [];
    vals = t[0].split(" ");    


    b = (no_trump and 1 or 2) - 1 ;
    e = trump and 5 or 1;

    for i in range(b*4,e*4):
        if(i % 2 == 1):
            continue;
        else:
            c = t[1][i];
            if c=="\n":
                continue
            arr = dict[c];
            outputs.append(arr);
        # if c=="\n":
        #     continue
        # arr = dict[c] * 1.0 / 14.0 + 0.5/14.0
        # outputs.append(arr);

    
    org_deals = [process_player(vals[i]) for i in range(0, 4)];
    #players = [[player[(i*13):((i+1):13)] for i in range(0, 4)] for player in deal]
    # return (deals, outputs);

    # no trump, spades, hearts, diamonds, clubs
    for suit in range(b, e):
        # south, east, north, west
        for vist in (0,2):
            deals = [numpy.copy(i) for i in org_deals]
            if suit > 1:
                for k in range(0, 4):
                    spades = numpy.copy(deals[k][0:13]);
                    # switch spades with current trump
                    deals[k][0:13] = deals[k][13*(suit - 1):13*suit];
                    deals[k][13*(suit-1):13*suit] = spades;
            current = deals[(4-vist):(len(deals))] + deals[0:(4-vist)];
            #current = numpy.concatenate(deals);

            data.append(numpy.concatenate(current));
            
    return (data, outputs); 

    

def process_player(cards):
    """
    Parse function used to process one players hand

    Parameters
    ----------
    cards : string
        A hand encoded in the aforementioned format

    Returns
    -------
    numpy.array
        Numpy array containing the encoded input

    """
    c = cards.split(".");
    p = [];

    # bit mapping for char values
    dict =     {
        'A' : 0,
        'K' : 1,
        'Q' : 2,
        'J' : 3,
        'T' : 4,
        '9' : 5,
        '8' : 6,
        '7' : 7,
        '6' : 8,
        '5' : 9,
        '4' : 10,
        '3' : 11,
        '2' : 12,
    }
    
    for t in c:
        v = numpy.repeat(0, 13);
        for i in t:
             v[dict[i]] = 1;

        p.append(v);
    # suits: Spades, Hearts, Diamonds, Clubs
    # contracts: None, Spades, Hearts, Diamonds, Clubs
    # east, north, west, south
    return numpy.concatenate(tuple(p));

def read_file(path, lines_count, shuffle = False, no_trump = True, trump = True, no_trump_test = True, trump_test = True, split = 0.66):
    """
    Function responsible for reading a whole file and generating datasets 

    Parameters
    ----------
    path : string
        File path
    lines_count : int
        Max lines count to be read
    shuffle : bool
        Boolean value indicating whether the lines are to be shuffled
    no_trump : bool
        Boolean value indicating whether No Trump deals are to be skipped
    trump : bool
        Boolean value indicating whether Trump deals are to be skipped

    Returns
    -------
    (list, list, list, list)
        Tuple of lists containing a set of generated inputs and a set of corresponding outputs (training and test data)

    """
    def process(data_set, outputs_set, line, no_trump, trump):
        data, outputs = parse(line, no_trump, trump);
        data_set.append(data)
        outputs_set.append(outputs)

    test_end = int(lines_count * split);
    data_set = []
    outputs_set = []
    test_set = []
    test_outputs_set = []
    line_number = 1
    lines = [];
    with open(path, "r") as file:
        for line in file:
            if line_number > lines_count:
                break
            #print(line)
            if(shuffle):
                lines.append(line)
            else:
                if line_number % 100  == 0:
                    print("Reading line {0}".format(line_number));
                if line_number < test_end:
                    process(data_set, outputs_set, line, no_trump, trump);
                else:
                    process(test_set, test_outputs_set, line, no_trump_test, trump_test);
            #data_set = data_set + data;
            #outputs_set = outputs_set + outputs;
            line_number = line_number + 1
    if(shuffle):
        random.shuffle(lines);
        line_number = 1;
        for line in lines:
            if line_number % 100  == 0:
                print("Reading line {0}".format(line_number));
            if line_number < test_end:
                process(data_set, outputs_set, line, no_trump, trump);
            else:
                process(test_set, test_outputs_set, line, no_trump_test, trump_test);
            line_number = line_number + 1;
    return combine_data_sets(data_set, outputs_set) + combine_data_sets(test_set, test_outputs_set);

def combine_data_sets(data_sets, output_sets):
    """
    Function used to flatmap a lists of lists

    Parameters
    ----------
    data_sets : list
        List of lists containing network inputs
    output_sets : list
        List of lists containing desired network outputs

    Returns
    -------
    (list, list)
        Tuple of lists containing a set of generated inputs and a set of corresponding outputs

    """
    data = []
    outputs = []
    for i in range(0, len(data_sets)):
        data_set = data_sets[i]
        outputSet = output_sets[i]
        if i % 100  == 0:
            print("appending {0}".format(i));
        for j in range(0, len(data_set)):
            data.append(data_set[j])
            outputs.append(outputSet[j])
    return  (data, outputs)

def save_as_tfrecord(data, output, name):
    """
    Function used to save a dataset as a TFRecord

    Parameters
    ----------
    data : list
        List of network inputs
    output : list
        List of containing desired network outputs

    """
    writer = tf.python_io.TFRecordWriter(name);
    for i in range(0, len(data)):
        inp = tf.train.Feature(float_list=tf.train.FloatList(value=data[i]));
        label = tf.train.Feature(float_list=tf.train.FloatList(value=output[i]));
        feature = {};
        feature['data'] = inp;
        feature['label'] = label;

        example = tf.train.Example(features=tf.train.Features(feature=feature));
        writer.write(example.SerializeToString());
    
    writer.close();

def divide_into_batches(batch_count, data_batches, outputs_batches, data, outputs):
    """
    Divides data into batches

    Parameters
    ----------
    batch_count : int
        Number of batches
    data_batches : list
        List of input data batches
    outputs_batches : list
        List of output data batches
    data : list
        All input data to be divided 
    outputs : int
        Output data to be batched
    """
    l = len(data);
    batch_size = int(l / batch_count);
    for i in range(0, batch_count-1):
        data_batches.append(data[i * batch_size : (i + 1) * batch_size]);
        outputs_batches.append(outputs[i * batch_size : (i + 1) * batch_size]);

    data_batches.append(data[(batch_count - 1) * batch_size : l]);
    outputs_batches.append(outputs[(batch_count - 1) * batch_size : l]);
        
def get_distribution(diffs):
    final = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for v in diffs:
        final[v] += 1
    return final;
    

def save_distribution(path, train_dist, test_dist):
    whole = path + "distributions.txt";
    os.makedirs(os.path.dirname(whole), exist_ok=True)
    f = open(whole, 'w');
    f.write("Train set")
    f.write(pprint.pformat(train_dist))

    f.write("Train set")
    f.write(pprint.pformat(test_dist))

    f.close();

def process_zipped_sets(zipped):
    samples_l = [];
    samples_r = [];
    outputs = [];
    diffs = [];
    for val in zipped:
        l = val[0];
        r = val[1];
        diffs.append(abs(l[1] - r[1]))
        samples_l.append(l[0]);
        samples_r.append(r[0]);
        if l[1] > r[1]:
            outputs.append(numpy.array([1, 0, 0]));
        elif l[1] < r[1]:
            outputs.append(numpy.array([0, 0, 1]));
        else:
            outputs.append(numpy.array([0, 1, 0]));

    return (samples_l, samples_r, outputs, diffs);

def generate_random_pairs(data, labels, count):
    input = list(zip(data, labels))
    left = random.sample(list(input), count);
    right = random.sample(input, count); 
    return process_zipped_sets(zip(left, right));


def generate_random_pair_for_samples(data, labels):
    count = len(data)
    input = list(zip(data, labels));
    right = random.sample(input, count); 
    return process_zipped_sets(zip(input, right));
      

def initialize_random(name):
    random.seed();
    s = random.getstate();
    with open(name, 'wb') as f:
        pickle.dump(s, f);

def reset_random(name):
    file = open(name, "rb");
    random.setstate(pickle.load(file));


def suit_count_for_params(no_trump, trump):
    acc = 0;
    if(no_trump):
        acc = acc + 1;
    if(trump):
        acc = acc + 4;
    return acc;

def generate_balanced_classes(data, labels, lefts, draws, rights):
    c = 0;
    input = list(zip(data, labels))
    samples_r = [];
    samples_l = [];
    outputs = [];
    diffs = [];
    while lefts > 0 or rights > 0 or draws > 0:
        if(c % 1000 == 0):
            print("Generated {0}".format(c))
        d = random.sample(input, 2);
        l = d[0];
        r = d[1];
        if l[1] > r[1] and lefts > 0:
            lefts -= 1;
            samples_l.append(l[0]);
            samples_r.append(r[0]);
            diffs.append(abs(l[1] - r[1]));
            outputs.append(numpy.array([1, 0, 0]));
            c += 1;
        elif l[1] < r[1] and rights > 0:
            rights -= 1;
            samples_l.append(l[0]);
            samples_r.append(r[0]);
            diffs.append(abs(l[1] - r[1]));
            outputs.append(numpy.array([0, 0, 1]));
            c += 1;
        else:
            if l[1] == r[1] and draws > 0:
                draws -= 1;
                samples_l.append(l[0]);
                samples_r.append(r[0]);
                diffs.append(abs(l[1] - r[1]));
                outputs.append(numpy.array([0, 1, 0]));
                c += 1;
    return (samples_l, samples_r, outputs, diffs);