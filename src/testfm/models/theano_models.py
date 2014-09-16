from testfm.models.cutil.interface import IModel

__author__ = 'linas'

'''
!!!EXPERIMENTAL!!!


Models using Theano. Adapted from: http://www.deeplearning.net/
'''

import math, time
import numpy
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import sparse
import logging
from scipy.sparse import dok_matrix
from testfm.models.cutil.interface import IModel


logger = logging.getLogger('testfm.models.theano_models')
logging.basicConfig()
logger.setLevel(logging.DEBUG)

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


class TheanoModel(IModel):
    ''' The main class for all of the theano based models that use some kind of neural network. '''

    def _convert(self, training_data):
        """
        Converts training_data pandas data frame into the RBM representation.
        The representation contains vector of movies for each user.
        """
        logger.debug('converting data of lenght {} to a sparse matrix and idexes'.format(len(training_data)))

        iid_map = {item: id for id, item in enumerate(training_data.item.unique())}
        uid_map = {user: id for id, user in enumerate(training_data.user.unique())}
        users = {user: set(entries) for user, entries in training_data.groupby('user')['item']}

        S = dok_matrix((len(uid_map), len(iid_map)), dtype=float)

        #train_set_x = numpy.zeros((len(uid_map), len(iid_map)))
        for user, items in users.items():
            for i in items:
                S[uid_map[user], iid_map[i]] = 1

        return S.tocsr(), uid_map, iid_map, users

    def get_score(self, user, item):

        #lets initialize visible layer to a user
        iid = self.iid_map[item]

        user_pred = self._get_user_predictions(user)
        return user_pred[0, iid]

class dA_CF(TheanoModel):

    '''
    We try to use autoencoders for CF. The idea that instead of making one kind of dimensionality
    reduction, we can do it with autoencoders.
    '''

    def __init__(self, n_hidden=400, learning_rate=0.8, training_epochs=15, corruption_level=0.1):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.corruption_level = corruption_level

    @classmethod
    def param_details(cls):
        return {
            "learning_rate": (0.01, 0.05, 0.5, 0.1),
            "training_epochs": (1, 50, 5, 15),
            "n_hidden": (10, 1000, 10, 100),
            "corruption_level" :(0.0, 1.0, 0.1, 0.1),
        }

    def set_params(self, n_hidden=100, learning_rate=0.1, training_epochs=5, corruption_level=0.1):
        """
        Set the parameters for the TensorCoFi
        """
        self.n_hidden = int(n_hidden)
        self.learning_rate = float(learning_rate)
        self.training_epochs = int(training_epochs)
        self.corruption_level = float(corruption_level)

    def fit(self,  training_data, batch_size=20, k=1, pretraining_epochs=5):

        index = T.lscalar()     # index to a [mini]batch
        x = T.matrix('x')       # the data is one row per user

        matrix, self.uid_map, self.iid_map, self.user_data = self._convert(training_data)
        training_set_x = theano.shared(matrix, name='training_data')
        n_train_batches = training_set_x.get_value(borrow=True).shape[0] / batch_size


        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.da = dA(numpy_rng=rng,
                    theano_rng=theano_rng,
                    input=x,
                    n_visible=training_set_x.get_value(borrow=True).shape[1],
                    n_hidden=self.n_hidden)

        cost, updates = self.da.get_cost_updates(
                    corruption_level=self.corruption_level,
                    learning_rate=self.learning_rate)

        train_da = theano.function([index], cost,
                    updates=updates,
                    givens={x: sparse.dense_from_sparse(training_set_x[index * batch_size: (index + 1) * batch_size])})

        for epoch in xrange(self.training_epochs):
            # go through trainng set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
            #print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    @lru_cache(maxsize=100)
    def _get_user_predictions(self, user):
        '''
        Compute the prediction for the user (predictions for all the items).
        It is cashed as we need to do it many times for evaluation.
        '''

        #print "computing prediction vector for user ",user

        user_items = self.user_data[user]

        matrix = numpy.zeros((1, len(self.iid_map))) #just one user for whoem we are making prediction

        #initialize the vector with items that user has experienced
        for i in user_items:
            matrix[0, self.iid_map[i]] = 1
        test_x = theano.shared(matrix, name='test_user')

        hidden = self.da.get_hidden_values(test_x)
        reconstructed = self.da.get_reconstructed_input(hidden)

        return reconstructed.eval()


class DBN_RBM_CF(TheanoModel):

    def __init__(self, hidden_layers_sizes=[500, 500], learning_rate=0.8, training_epochs=7):
        self.hidden_layers_sizes = hidden_layers_sizes
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs

    def get_name(self):
        return "DBN:RBM (n_hidden={0}, learning_rate={1}, training_epochs={2})"\
            .format(self.hidden_layers_sizes, self.learning_rate, self.training_epochs)


    def fit(self, training_data, batch_size=20, k=1, pretraining_epochs=5):

        matrix, uid_map, iid_map, user_data = self._convert(training_data)

        self.user_data = user_data
        self.uid_map = uid_map
        self.iid_map = iid_map

        training_set_x = theano.shared(matrix, name='training_data')
        n_train_batches = training_set_x.get_value(borrow=True).shape[0] / batch_size

        #print '... getting the pretraining functions'
        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.dbn = DBN(n_ins=training_set_x.get_value(borrow=True).shape[1],
                       numpy_rng=rng,
                       theano_rng=theano_rng,
                       hidden_layers_sizes=self.hidden_layers_sizes)
        pretraining_fns = self.dbn.pretraining_functions(train_set_x=training_set_x, batch_size=batch_size, k=k)

        ## Pre-train layer-wise
        for i in xrange(self.dbn.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index, lr=self.learning_rate))
                #print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                #print numpy.mean(c)

    @lru_cache(maxsize=100)
    def _get_user_predictions(self, user):
        '''
        Compute the prediction for the user (predictions for all the items).
        It is cashed as we need to do it many times for evaluation.

        We go from input units till the top of the DBN and back.
        '''

        user_items = self.user_data[user]

        matrix = numpy.zeros((1, len(self.iid_map))) #just one user for whoem we are making prediction

        #initialize the vector with items that user has experienced
        for i in user_items:
            matrix[0, self.iid_map[i]] = 1
        test_x = theano.shared(matrix, name='user-model')

        for rbm in self.dbn.rbm_layers:
            _, test_x = rbm.propup(test_x)
        for rbm in reversed(self.dbn.rbm_layers):
            _, test_x = rbm.propdown(test_x)
        return test_x.eval()




class RBM_CF(TheanoModel):
    """
    Restricted Boltzmann Machines (RBM) is used in the CF as one of the most successful model
    for collaborative filtering:

    Ruslan Salakhutdinov, Andriy Mnih, Geoffrey E. Hinton:
        Restricted Boltzmann machines for collaborative filtering. ICML 2007: 791-798

    The idea, is that you represent each user by vector of watched/unwatched movies.
    You train the RBM using such data, i.e., each user is an example for RBM.
    So RBM has n_visible equal to the #movies. The number of hidden units is a hyper-parameter.

    You make predictions for a user by providing a vector of
    his movies and doing gibbs sampling via visible-hidden-visible states.
    The activation levels on visible will be your predictions.
    """

    def __init__(self, n_hidden=400, learning_rate=0.8, training_epochs=7):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs

    @classmethod
    def param_details(cls):
        return {
            "learning_rate": (0.01, 0.05, 0.5, 0.1),
            "training_epochs": (1, 50, 5, 15),
            "n_hidden": (10, 1000, 10, 100),
        }

    def set_params(self, n_hidden=100, learning_rate=0.1, training_epochs=5):
        """
        Set the parameters for the TensorCoFi
        """
        self.n_hidden = int(n_hidden)
        self.learning_rate = float(learning_rate)
        self.training_epochs = int(training_epochs)

    def get_name(self):
        return "RBM (n_hidden={0}, learning_rate={1}, training_epochs={2})"\
            .format(self.n_hidden, self.learning_rate, self.training_epochs)


    def fit(self, training_data):
        '''
        Fits the RBM using training data.
        '''

        csr_matrix, uid_map, iid_map, user_data = self._convert(training_data)
        self.user_data = user_data
        self.uid_map = uid_map
        self.iid_map = iid_map

        training_set_x = theano.shared(csr_matrix, name='training_data')
        self.train_rbm(training_set_x)

    def train_rbm(self, train_set_x, batch_size=20):
        """
        Trains the RBM, given the training data set.

        :param train_set_x: a matrix of user vectors for trainign RBM
        :param learning_rate: learning rate used for training the RBM
        :param training_epochs: number of epochs used for training
        :param batch_size: size of a batch used to train the RBM
        :param n_chains: number of parallel Gibbs chains to be used for sampling
        :param n_samples: number of samples to plot for each chain

        """


        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        logger.debug('traing_rbm: #batches:{}, batch_size:{}'.format(n_train_batches, batch_size))
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as user vector per each row

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        # initialize storage for the persistent chain (state = hidden layer of chain)
        persistent_chain = theano.shared(numpy.zeros((batch_size, self.n_hidden),
                                                     dtype=theano.config.floatX),
                                         borrow=True)

        # construct the RBM class
        logger.debug('Constructing RBM')
        self.rbm = RBM(n_visible=train_set_x.get_value(borrow=True).shape[1],
                       n_hidden=self.n_hidden,
                       input=x,
                       numpy_rng=rng,
                       theano_rng=theano_rng)


        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = self.rbm.get_cost_updates(lr=self.learning_rate, persistent=persistent_chain, k=self.training_epochs)

        #################################
        #     Training the RBM          #
        #################################
        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        logger.debug('Creating training function for rbm')
        train_rbm = theano.function([index],
                                    cost,
                                    updates=updates,
                                    givens={x: sparse.dense_from_sparse(train_set_x[index * batch_size: (index + 1) * batch_size])},
                                    name='train_rbm')

        logger.debug('start training')
        # go through training epochs
        for epoch in xrange(self.training_epochs):
            # go through the training set
            mean_cost = []
            for batch_index in xrange(n_train_batches):
                mean_cost += [train_rbm(batch_index)]
            logger.debug('Training epoch {}, cost is {}'.format(epoch, numpy.mean(mean_cost)))

    @lru_cache(maxsize=100)
    def _get_user_predictions(self, user):
        '''
        Compute the prediction for the user (predictions for all the items).
        It is cashed as we need to do it many times for evaluation.
        '''

        #print "computing prediction vector for user ",user

        user_items = self.user_data[user]

        matrix = numpy.zeros((1, len(self.iid_map))) #just one user for whoem we are making prediction

        #initialize the vector with items that user has experienced
        for i in user_items:
            matrix[0, self.iid_map[i]] = 1
        test_x = theano.shared(matrix, name='test_user')

        #number_of_times_up_down = 100
        # define one step of Gibbs sampling (mf = mean-field) define a
        presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples = self.rbm.gibbs_vhv(test_x)

        return vis_mfs.eval()

class RBM(object):
    """
    The implementation of RBM taken from http://www.deeplearning.net/tutorial/rbm.html

    """
    def __init__(self, n_visible, n_hidden=100, input=None, learning_rate=0.1, training_epochs=5,\
                 W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        #the problem that we don't need this parameter as it will be set automatically
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs



        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=float(-4.0 * math.sqrt(6.0 / float(n_hidden + n_visible))),
                      high=float(4.0 * math.sqrt(6.0 / float(n_hidden + n_visible))),
                      size=(n_visible, n_hidden)),
                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)
        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

        #output is computed on the fly, without storing it as real hidden units
        self.output = self.propup(input)[1]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                    dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))

        return cross_entropy


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output.

    We use only unsupervised part of DBN, i.e., we do not use the logistic regression
    layer on the top. So we take a user, go all the way up and then we go down to
    the input layer to generate the prediction.

    """

    def __init__(self, n_ins, numpy_rng, theano_rng=None,  hidden_layers_sizes=[500, 500]):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.rbm_layers[i - 1].output

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i])
            self.rbm_layers.append(rbm_layer)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: sparse.dense_from_sparse(train_set_x[batch_begin:batch_end])})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns


class dA(object):
    """
    Taken form theano:
    Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
