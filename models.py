import tensorflow as tf;
import numpy as np;
import math;
import data_parser as dp;

class Model:
    """
    Base class for models containing static methods

    """
    def mse_loss(pred, actual):
        """
        Mean Squared Error loss function 

        Parameters
        ----------
        pred : Tensor
            Tensor containing the network's output 
        actual : Autoencoder
            Tensor containing the desired output

        Returns
        -------
        Tensor 
            Tensor used to calculate the error's value
        """
        return tf.reduce_mean(tf.pow(actual - pred, 2));

    def cross_entropy_loss(pred, actual):
        """
        Cross Entropy loss function 

        Parameters
        ----------
        pred : Tensor
            Tensor containing the network's output 
        actual : Autoencoder
            Tensor containing the desired output

        Returns
        -------
        Tensor 
            Tensor used to calculate the error's value
        """
        p = tf.convert_to_tensor(pred);
        a = tf.convert_to_tensor(actual);
        crossEntropy = tf.add(tf.multiply(tf.log(p + 1e-10), a), tf.multiply(tf.log(1 - p + 1e-10), 1 - a));
        return -tf.reduce_mean(tf.reduce_sum(crossEntropy, 1));


    def direct_error(pred, actual):
        """
        Direct loss function 

        Parameters
        ----------
        pred : Tensor
            Tensor containing the network's output 
        actual : Autoencoder
            Tensor containing the desired output

        Returns
        -------
        Tensor 
            Tensor used to calculate the error's value
        """
        return tf.reduce_mean(tf.abs(actual - pred));

    def initialize_optimizer(opt, vars):
        """
        Initializes all optimizer slots

        Parameters
        ----------
        opt : Optimizer
            Optimizer that needs initialization
        vars : Autoencoder
            Variables used

        Returns
        -------
        Operation 
            Initialization operation
        """
        to_init = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in vars];
        if(type(opt) is tf.train.AdamOptimizer):
            to_init.extend(s for s in list(opt._get_beta_accumulators()) if s is not None);    
        
        return tf.variables_initializer([s for s in to_init if s is not None]);

class Autoencoder(Model):
    """
    Class implementing a multilayered Autoencoder Network
    Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
    """


    @property
    def session(self):
        """
        Model's Tensorflow session

        Parameters
        ----------
        self : Autoencoder

        Returns
        -------
        tensorflow.Session 
            Session currently used by the model
        """
        return self.__session;

    def create_layer(self, index, input, is_fixed = False, is_decoder = False):
        """
        Creates an autoencoder layer

        Parameters
        ----------
        self : Autoencoder
        index : int
            Layer index
        input : Tensor
            Previous layer feed to the new layer
        is_fixed : bool
            Boolean value indicating whether the layer needs to be temporarily frozen
        is_decoder : bool
            Boolean value indicating if the layer needs to serve as an encoder or a decoder
        Returns
        -------
        Tensor 
            Created layer
        """
        if is_fixed:
            if is_decoder:
                return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixed_weights[index], transpose_b = is_decoder), self.out_biases_fixed[index]));
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.fixed_weights[index], transpose_b = is_decoder), self.fixed_biases[index]));
        if is_decoder:
            return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = is_decoder), self.out_biases[index]));
        return tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights[index], transpose_b = is_decoder), self.biases[index]));

    def create_weights(self, prev_count, curr_count):
        """
        Creates weights for a layer

        Parameters
        ----------
        self : Autoencoder
        prev_count : int
            Neuron count of the previous layer
        curr_count : Tensor
            Neuron count of the current layer

        """
        w = tf.Variable(tf.random_normal([prev_count, curr_count]), trainable = True, name='v_W{0}'.format(curr_count));
        b = tf.Variable(tf.random_normal([curr_count]), trainable = True, name='v_B{0}'.format(curr_count));
        self.weights.append(w);
        self.biases.append(b);
        w_f = tf.Variable(tf.identity(w), trainable = False, name='f_W{0}'.format(curr_count));
        b_f = tf.Variable(tf.identity(b), trainable = False, name='f_B{0}'.format(curr_count));
        self.fixed_weights.append(w_f);
        self.fixed_biases.append(b_f);
        
        b_out = tf.Variable(tf.random_normal([prev_count]), trainable = True, name='v_B_out{0}'.format(curr_count));
        b_out_fixed = tf.Variable(tf.identity(b_out), trainable = False, name='f_B_out{0}'.format(curr_count));
        self.out_biases.append(b_out);
        self.out_biases_fixed.append(b_out_fixed);

    def clone_weights(self, a):

        # copy weights
        for w in a.weights:
            _w = tf.Variable(w);
            self.weights.append(_w);
            self.fixed_weights.append(tf.Variable(tf.identity(_w)))

        # copy biases
        for b in a.biases:
            _b = tf.Variable(b);
            self.biases.append(_b);
            self.fixed_biases.append(tf.Variable(tf.identity(_b)));

        # copy output biases
        for b_out in a.out_biases:
            _b_out = tf.Variable(b_out);
            self.out_biases.append(_b_out);
            self.out_biases_fixed.append(tf.Variable(tf.identity(_b_out)))




    @classmethod
    def clone(cls, src):
        """
        Copy constructor
        :param a:
        :param autoencoder:
        """
        a = cls(src.input_count, src.layer_counts, src.loss)
        a.clone_weights(src)
        a.__session = src.session;
        a.session.run(tf.variables_initializer(a.weights + a.biases + a.fixed_biases + a.fixed_weights + a.out_biases + a.out_biases_fixed))
        return a;

    @classmethod
    def build(cls, input_count, layer_counts, loss):
        a = cls(input_count, layer_counts, loss);
        a.prepare_session();
        for i in range(0, len(a.layer_counts)-1):
            a.create_weights(a.layer_counts[i], a.layer_counts[i + 1]);

        init = tf.global_variables_initializer();
        a.session.run(init);
        return a;

    def __init__(self, input_count, layer_counts, loss):
        """
        Main class constructor

        Parameters
        ----------
        self : Autoencoder
        input_count : int
            Number of network's inputs
        layer_counts : list
            List of neuron counts for each layer
        loss : Tensor
            Loss function used during model optimalization

        """
        self.loss = loss;
        self.input_count = input_count;
        self.layer_counts = [input_count] +layer_counts;
        self.weights = [];
        self.biases = [];
        self.out_biases = [];
        self.out_biases_fixed = [];
        self.input = tf.placeholder("float", [None, self.input_count]);
        self.fixed_weights = [];
        self.fixed_biases = []
        l = len(layer_counts);
        self.__session = None


    def get_variables_to_init(self, n):
        """
        Creates a list of variables that need to be currently initialized

        Parameters
        ----------
        self : Autoencoder
        n : int
            Layer index

        Returns
        -------
        list 
            Returns a list of tensorflow.Variable objects that neeed to be intialized during this step

        """

        vars = [];

        if 0<n:
            vars.append(self.fixed_biases[n-1]);
            vars.append(self.fixed_weights[n-1]);
            vars.append(self.out_biases_fixed[n-1]);
        return vars;

    def prepare_session(self):
        """
        Prepares a new session for the model

        Parameters
        ----------
        self : Autoencoder

        """
        config = tf.ConfigProto();
        self.__session = tf.Session(config=config);

     
    def pretrain(self, learning_rate, i, it, data, ep, delta, summary_path, optimizer_class = tf.train.RMSPropOptimizer, m = 0.2, decay = 0.9, no_improvement = 5):
        """
        Pretrains one layer with specified parameters
        Please remember that the summary_path must contain one argument for formatting
        Parameters
        ----------
        self : Autoencoder
        learning_rate : int
            Learning rate for the optimizer
        i : int
            Layer index
        it : int
            Number of iterations used
        data : list
            List of numpy arrays feed to the input placeholder used for training
        delta : float
            If the improvement between epochs is smaller than delta, the training process is aborted
        ep : float
            If it is 0, then the training will be executed until the error value is larger than ep
        summary_path : string
            Path used to store Tensorflow summaries generated by the model during training
        optimizer_class : float
            Optimized class used during pretraining
        m : float
            Momentum
        decay : float
            Learning rate decay
        no_improvement : int
            If the error function value worsens the amount of times specified by this parameter the calculation will be aborted
        """

        input = self.input;
        step = tf.Variable(0, name='global_step', trainable=False);
        net = self.build_pretrain_net(i, input);
        loss_function = self.loss(net[len(net) - 1], input);
        if(optimizer_class is tf.train.GradientDescentOptimizer):
            opt = optimizer_class(learning_rate);
        else:
            opt = optimizer_class(learning_rate, momentum = m);
        optimizer = opt.minimize(loss_function, global_step=step);    
        vars = self.get_variables_to_init(i);
        vars.append(step);
        self.session.run(tf.variables_initializer(vars));  
        vars.extend([self.weights[i], self.biases[i], self.out_biases[i]])
        self.session.run(Model.initialize_optimizer(opt, vars));
        loss_summary = tf.summary.scalar("loss", loss_function);
        weights_summary = tf.summary.histogram("weights", self.weights[i]);
        biases_summary = tf.summary.histogram("biases", self.biases[i]);
        summary_op = tf.summary.merge([loss_summary, weights_summary, biases_summary]);
        writer = tf.summary.FileWriter(summary_path.format(i) + (ep > 0 and "ep{0}".format(ep) or "it{0}".format(it)) , graph=self.session.graph, flush_secs = 10000);

        if(delta > 0):
            no_improvement_counter = 0;
            prev_val = 0;
            it_counter = 0;
            while True:
                for k in range(0, len(data)):
                    lval, _, summary = self.session.run([loss_function, optimizer, summary_op], feed_dict={input : data[k]});
                    if it_counter % 100 == 0:
                        print("pretraining {0} - it {1} - lval {2}".format(i, it_counter, lval));
                        writer.add_summary(summary, it_counter);
                        # no significant change
                        if prev_val != 0 and (prev_val - lval) < delta:
                            if(no_improvement_counter > no_improvement):
                                print("terminating due to no improvement");
                                print("pretraining {0} - it {1} - lval {2}".format(i, it_counter, lval));
                                return
                            else:
                                no_improvement_counter = no_improvement_counter + 1;
                        prev_val = lval;
                    it_counter = it_counter + 1;


        elif it > 0:
            for j in range(1, it):
                    for k in range(0, len(data)):
                        lval, _, summary = self.session.run([loss_function, optimizer, summary_op], feed_dict={input : data[k]});
                    if j % 100 == 0:
                        print("pretraining {0} - it {1} - lval {2}".format(i, j, lval));
                        writer.add_summary(summary, j);
        else:
            j = 0;
            while True:
                for k in range(0, len(data)):
                    _, summary, lval = self.session.run([optimizer, summary_op, loss_function], feed_dict={input : data[k]});
                
                if j % 100 == 0:
                    print("pretraining {0} - it {1} - lval {2}".format(i, j, lval));
                    writer.add_summary(summary, j);
                j = j + 1;
                if(lval <= ep):
                    print("pretraining ended {0} - it {1} - lval {2}".format(i, j, lval));
                    break;



            
    def build_complete_net(self, input):
        """
        Builds a complete network of all encoding layers 

        Parameters
        ----------
        self : Autoencoder
        input : Tensor
            Input placeholder used to feed data to the network

        Returns
        -------
        list
            List of Tensor objects that create the network

        """
        net = [];
        inp = input;
        for i in range(0, len(self.weights)):
            inp = self.create_layer(i, inp);
            net.append(inp);
            
        return net;

    
    def build_pretrain_net(self, n, input):
        """
        Builds a partially frozen and uncomplete network used for pretraining steps

        Parameters
        ----------
        self : Autoencoder
        n : int
            Layer index
        input : Tensor
            Input placeholder used to feed data to the network

        Returns
        -------
        list
            List of Tensor objects that create the network
            
        """
        layers = [];
        inp = input;
        for i in range(0, n):
            inp = self.create_layer(i, inp, is_fixed = True);
            layers.append(inp);
        
        inp = self.create_layer(n, inp);
        layers.append(inp);

        inp = self.create_layer(n, inp, is_decoder = True);
        layers.append(inp);
        
        for i in range(0, n):
            inp = self.create_layer(n - 1 - i, inp, is_fixed = True, is_decoder = True);
            layers.append(inp);
        return layers;

class Classifier(Model):
    """
    Class used to append a classifier to a pretrained autoencoder

    """
    autoencoder_r: Autoencoder
    autoencoder_l: Autoencoder

    def __init__(self, autoencoder, outputs):
        """
        Class constructor

        Parameters
        ----------
        self : Classifier 
        autoencoder : Autoencoder
            The autoencoder object to which the classifier will be appended to

        """
        # store encoders
        self.autoencoder_l = autoencoder;
        self.autoencoder_r = Autoencoder.clone(autoencoder);
        # initialize placeholders
        self.input_placeholder_l = tf.placeholder("float", [None, self.autoencoder_l.input_count]);
        self.input_placeholder_r = tf.placeholder("float", [None, self.autoencoder_r.input_count]);
        # build final encoding layers
        self.encoder_l = self.autoencoder_l.build_complete_net(self.input_placeholder_l);
        self.encoder_r = self.autoencoder_r.build_complete_net(self.input_placeholder_r);

        input_l = self.encoder_l[len(self.encoder_l) - 1];
        input_r = self.encoder_r[len(self.encoder_r) - 1];

        input = tf.concat([input_l, input_r], 1)
        self.session = autoencoder.session;

        self.weights = tf.Variable(tf.random_normal([input.shape[1].value, outputs]));
        self.biases = tf.Variable(tf.random_normal([outputs]));
        self.layer = tf.nn.softmax(tf.matmul(input, self.weights) + self.biases);
        self.output_placeholder = tf.placeholder("float", [None, outputs]);
        self.get_accuracy_tensors();
        
    def create_train_summary(self, data_l, data_r, output, diffs, test_data_l, test_data_r, test_output, test_diffs, res):

        def draw_whole_set(s, res, type):
            s.value.add(tag="{0} accuracy".format(type), simple_value=res[0])
            s.value.add(tag="{0} - Precision - Draws".format(type), simple_value=res[1])
            s.value.add(tag="{0} - Recall - Draws".format(type), simple_value=res[2])
            # draw corrects
            s.value.add(tag="{0} - Precision - Nondraws".format(type), simple_value=res[3])
            s.value.add(tag="{0} - Recall - Nondraws".format(type), simple_value=res[4])
            s.value.add(tag="{0} - Acc TF".format(type), simple_value=res[5])
    

            return s;

    
        s = tf.Summary();
        r_tr = self.test(data_l, data_r, output, diffs);
        r_test = self.test(test_data_l, test_data_r, test_output, test_diffs);
        s = draw_whole_set(s, r_tr, "Train");
        s = draw_whole_set(s, r_test, "Test");
        correct_correct, correct_wrong, wrong_correct, wrong_wrong = res
        ## TO DO: stats
        s.value.add(tag="correct confirmed", simple_value=(float(correct_correct)/float(correct_correct + correct_wrong)))
        s.value.add(tag="correct failed", simple_value=(float(correct_wrong)/float(correct_correct + correct_wrong)))
        s.value.add(tag="wrong improved", simple_value=(float(wrong_correct)/float(wrong_correct + wrong_wrong)))
        s.value.add(tag="still wrong", simple_value=(float(Wrong_wrong)/float(wrong_wrong + wrong_correct)))


        return s;

    def train(self, data_l, data_r, desired_output, learning_rate, it, delta, path, train_data_l, train_data_r, train_output, diffs, test_data_l, test_data_r, test_output, test_diffs, test_data, test_labels, net_outputs, comparables, train_suits = 5, test_suits = 5, loss_f = Model.mse_loss, no_improvement = 5, experiment_name = ""):
        """
        Main train method
    
        This method is used to start the fine tuning phase of the classifier and the autoencoder layers

        Parameters
        ----------
        self : Classifier 
        data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        desired_output : list
            List of numpy arrays with labels for the corresponding training inputs
        learning_rate : float
            Learning rate for the optimizer used
        it : int
            Iterations count to be executed
        delta : float
            If the improvement between epochs is smaller than delta, the training process is aborted
        path : string
            Path used to store summaries generated by Tensorflow
        test_data : list 
            Test inputs used for validation
        test_output : list
            Test outputs used for validation
        loss_f : Tensor
            Loss function used by the optimizer
        no_improvement : int
            If the error function value worsens the amount of times specified by this parameter the calculation will be aborted
            
        """
        loss = loss_f(self.output_placeholder, self.layer);
        opt = tf.train.RMSPropOptimizer(learning_rate);
        optimizer = opt.minimize(loss);
        self.session.run(tf.variables_initializer([self.weights, self.biases]));
        slot_vars = [self.weights, self.biases] + self.autoencoder_r.weights+ self.autoencoder_l.weights + self.autoencoder_l.biases + self.autoencoder_r.biases;
        self.session.run(Model.initialize_optimizer(opt, slot_vars));
        #self.session.run(tf.initialize_all_variables());
        hist_summaries = [(self.autoencoder_l.weights[i], 'weights{0}'.format(i)) for i in range(0, len(self.autoencoder_l.weights))];
        hist_summaries.extend([(self.autoencoder_l.biases[i], 'biases{0}'.format(i)) for i in range(0, len(self.autoencoder_l.weights))]);
        summaries = [tf.summary.histogram(v[1], v[0]) for v in hist_summaries];
        summaries.append(tf.summary.scalar("loss_final", loss));
        summary_op = tf.summary.merge(summaries);
        writer = tf.summary.FileWriter(path, graph=self.autoencoder_l.session.graph)


        if delta > 0:
            prev_val = 0;
            current_val = 0;
            no_improvement_counter = 0;
            it_counter = 0;
            while True:
                for k in range(0, len(data_r)):
                    lval, _, summary = self.session.run([loss, optimizer, summary_op], feed_dict={self.input_placeholder_l: data_l[k], self.input_placeholder_r: data_r[k], self.output_placeholder: desired_output[k]});
                if it_counter % 1000 == 0:
                    res = self.classify_sequential(test_data, comparables, test_labels, net_outputs)
                    s = self.create_train_summary(train_data_l, train_data_r, train_output, diffs, test_data_l, test_data_r, test_output, test_diffs, res);
                    #current_val = self.test(test_data_l, test_data_r, test_output)[0];
                    #print(current_val);
                    self.save_model(experiment_name + " at {0}".format(it_counter))
                    print("finetuning - it {0} - lval {1}".format(it_counter, lval));
                    writer.add_summary(summary, it_counter);
                    writer.add_summary(s, it_counter);
                    if prev_val != 0 and (current_val - prev_val) < delta:
                        print(current_val - prev_val);
                        if(no_improvement_counter > no_improvement):
                            print("terminating due to no improvement");
                            print("finetuning - it {0} - lval {1}".format(it_counter, lval));
                            return
                        else:
                            no_improvement_counter = no_improvement_counter + 1;
                    prev_val = current_val;
                it_counter = it_counter + 1;

        else:
            for i in range(0, it):
                for k in range(0, len(data)):
                    lval, _, summary = self.autoencoder.session.run([loss, optimizer, summary_op], feed_dict={self.input_placeholder_l: data_l[k], self.input_placeholder_r: data_r,  self.output_placeholder: desired_output[k]});
                if i % 100 == 0:
                    print("finetuning - it {0} - lval {1}".format(i, lval));
                    writer.add_summary(summary, i);

    def get_accuracy_tensors(self):
        """
        Method used to set up tensors holding accuracy values

        Parameters
        ----------
        self : Classifier 
        data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        desired_output : list
            List of numpy arrays with labels for the corresponding training inputs
            
        Returns
        -------
        (Tensor, Tensor, Tensor)
            Tuple containing all tensors hold accuracy values
        """
        correct_prediction = tf.equal(tf.argmax(self.layer, 1), tf.argmax(self.output_placeholder, 1));

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.accuracy;
        
    def test(self, data_l, data_r, desired_output, diffs, margin = 0.2):
        """
        Test method
    
        This method is used to obtain the accurancy of the created model including information such as:
            - Exact deal match percentage
            - Actual deal missed by 1 percentage
            - Actual deal missed by 2 percentage

        Parameters
        ----------
        self : Classifier 
        data : list
            List of numpy arrays with training inputs which will be fed to the input placeholder
        desired_output : list
            List of numpy arrays with labels for the corresponding training inputs
            
        Returns
        -------
        Detailed accurancy concerning the current model
        """
        res = [];
        [output, acc] = self.session.run([self.layer, self.accuracy], feed_dict={self.input_placeholder_l: data_l, self.input_placeholder_r: data_r, self.output_placeholder: desired_output});
        l = len(diffs)
        tp_nondraws = 0;
        tp_draw = 0
        fn_nondraws = 0;
        fn_draw = 0
        fp_nondraws = 0;
        fp_draw = 0
        global_correct = 0;
        for i in range(0,l):
            r = diffs[i]
            # check if sample is marked as a draw
            if(output[i][0] < output[i][1]):
                # marked correctly
                if(r == 0):
                    tp_draw += 1;
                    global_correct += 1;
                else:
                    fp_draw += 1;
                    fn_nondraws += 1;
            # is not marked as a draw
            else:
                if(r == 0):
                    fp_nondraws += 1;
                    fn_draw += 1;
                else:
                    tp_nondraws += 1;
                    global_correct += 1;


        
        # write accuracy
        res.append(float(global_correct)/float(len(diffs)))
        res.append(float(tp_draw) / (float(tp_draw + fp_draw) or 1))
        res.append(float(tp_draw) / (float(tp_draw + fn_draw) or 1))
        res.append(float(tp_nondraws) / (float(tp_nondraws + fp_nondraws) or 1))
        res.append(float(tp_nondraws) / (float(tp_nondraws + fn_nondraws) or 1))
        res.append(acc)


        return res;          
            

    def save_model(self, name):
        """
        Save method

        This method stores the current model (including the autoencoder it contains) on permament memory 
        under the location ./name .

        Parameters
        ----------
        self : Classifier 
        name : string
            Model filename
        test_labels : list
            List of numpy arrays with labels for the corresponding training inputs
        suits : int
            A number indicating the number of suits in the input data    
        """
        saver = tf.train.Saver();
        saver.save(self.autoencoder_l.session, "./models/{0}".format(name));

    def restore_model(self, name):
        """
        Restore method

        This method restores a before saved model (including the autoencoder it contains) from permament memory. 
        Please remember to setup as instance of the Classifier and Autoencoder classes before running it, as well
        as starting a Tensorflow session.

        Parameters
        ----------
        self : Classifier 
        name : string
            Model filename
        """
        saver = tf.train.Saver();
        saver.restore(self.autoencoder_l.session, "./models/{0}".format(name));

    def multi_batch_test(self, suits, data_batches, outputs_batches, batch_count):
        """
        Multibatch testing method

        This method aggregates the results for several batches.

        Parameters
        ----------
        self : Classifier 
        suits : int
            Number of suits
        data_batches : list
            List of input data batches
        outputs_batches : list
            List of output data batches
        batch_count : int
            Number of batches
        """
        res = [];
        for i in range(0, batch_count):
            res.append(self.suit_based_accurancy(data_batches[i], outputs_batches[i], suits));

        return ([sum(z)/batch_count for x in list(zip(*res)) for z in zip(*x)], res);

    # def compare_sequential_bulk(self, samples, comparables, outputs, net_outputs, margin):
    #     l = len(samples);
    #     totals = [];
    #     for i in range(0, 14):
    #         current = [[0,0]] * l;
    #         for j in range(0, len(comparables[i])):
    #             # construct answers
    #             testing = [comparables[i][j]] * l; 
    #             desired_outputs = map(lambda x: dp.get_output_for_pair(x, i), outputs);
    #             o = self.session.run(self.layer, feed_dict={self.input_placeholder_l: samples, self.input_placeholder_r: testing, self.output_placeholder: desired_output});
    #             current = map(lambda x: list(map(sum, zip(*x))), zip(o, current));

    #         totals.append(current);
        
    def classify_sequential(self, samples, comparables, outputs, net_outputs, margin):
        correct_correct = 0;
        correct_wrong = 0;
        wrong_wrong = 0;
        wrong_correct = 0;
        for i in range(0, len(samples)):
            ans = self.compare_sequential(sample[i], comparables, outputs[i], margin);
            # current value is right
            if(outputs[i] == ans):
                # network was also right
                if(outputs[i] == net_outputs[i]):
                    correct_correct += 1;
                # network was wrong
                else:
                    wrong_correct += 1;
            # current value is wrong
            else:
                # network was right before
                if(outputs[i] == net_outputs[i]):
                    correct_wrong += 1;
                # both models were wrong
                else:
                    wrong_wrong += 1;

        return (correct_correct, correct_wrong, wrong_correct, wrong_wrong);





    def compare_sequential(self, sample, comparables, output, margin):
        for i in range(0, 14):
            left = 0;
            right = 0;
            l = len(comparables[i])
            samples = [sample] * l;
            # construct answers
            testing = comparables[i]; 
            desired_outputs = [dp.get_output_for_pair(output, i)] * l;
            o = self.session.run(self.layer, feed_dict={self.input_placeholder_l: samples, self.input_placeholder_r: testing, self.output_placeholder: desired_output});
            for val in o:
                if math.abs(val[0] - val[1]) < margin:
                    # this is a draw
                    return i;
                elif val[0] > val[1]:
                    left += 1;
                else:
                    right += 1;
            if(left < right):
                return i;
        return 0;
            
            

        
        
