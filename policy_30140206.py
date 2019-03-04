from policies import base_policy as bp
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle


# program consts:
# INPUT_SIZE = 12
OUTPUT_SIZE = 3
PLAYER_STATE_SIZE = 6
BATCH_SIZE = 96
ARRAY_SIZE = 3*BATCH_SIZE
WINDOW_SIZE = 7
HIDEN_SIZE = 10
HIDEN_SIZE2 = 8
NEAREST_AREA = 3
ACTION_TO_INDEX = {'CC':0, 'CW':1, 'CN':2}
IMITATION_ROUND_NUMBER = 300
FIRST_CONVOLUTION_WINDOW = 3
SECOND_CONVOLUTION_WINDOW = 3
EPSILON_DECREASE_RATE = 0.7
EPSILON_MINIMUM_VALUE = 0.05

class Policy30140206(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        self.state = np.zeros(100)
        input_size = WINDOW_SIZE * WINDOW_SIZE

        # creating the network for deep Q learning
        tf.reset_default_graph()
        self.inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32)

        # getting parameters from pickle
        try:
            state = pickle.load(open(self.load_from, 'rb'))
            # load cnn variables:
            # first convolutional layer:
            self.W_conv1 = tf.Variable(tf.constant(state[0]))
            self.b_conv1 = tf.Variable(tf.constant(state[1]))

            # scond convolutional layer:
            self.W_conv2 = tf.Variable(tf.constant(state[2]))
            self.b_conv2 = tf.Variable(tf.constant(state[3]))


            # Densely Connected Layer (flat connection):
            self.W_fc1 = tf.Variable(tf.constant(state[4]))
            self.b_fc1 = tf.Variable(tf.constant(state[5]))

            #last conection to output:
            self.W_fc2 = tf.Variable(tf.constant(state[6]))
            self.b_fc2 = tf.Variable(tf.constant(state[7]))

            self.actions_count = state[8]

        #     data for supervised learning:
            self.rewardDict = state[9]
            self.lastObject = state[10]

            # data structures for remember the history of game, for learning:
            self.statesList = state[11]
            self.rewardsList = state[12]
            self.QtargetsList = state[13]
            self.actionsList = state[14]

        except:
            # init cnn variables:
            # first convolutional layer:
            self.W_conv1 = self.weight_variable([FIRST_CONVOLUTION_WINDOW, FIRST_CONVOLUTION_WINDOW, 1, 5])
            self.b_conv1 = self.bias_variable([5])

            # scond convolutional layer:
            self.W_conv2 = self.weight_variable([SECOND_CONVOLUTION_WINDOW, SECOND_CONVOLUTION_WINDOW, 5, 10])
            self.b_conv2 = self.bias_variable([10])

            # Densely Connected Layer:
            self.W_fc1 = self.weight_variable([WINDOW_SIZE * WINDOW_SIZE * 10, 100])
            self.b_fc1 = self.bias_variable([100])
            self.actions_count = -1

            #last conection to output:
            self.W_fc2 = self.weight_variable([100, 3])
            self.b_fc2 = self.bias_variable([3])

            # data structures for remember the history of game, for learning:
            self.statesList = [None] * ARRAY_SIZE
            self.rewardsList = [None] * ARRAY_SIZE
            self.QtargetsList = [None] * ARRAY_SIZE
            self.actionsList = [None] * ARRAY_SIZE

        # convert the input layer to 2D matrix, and conenct the CNN layers:
        input_2D = tf.reshape(self.inputs, [-1, WINDOW_SIZE, WINDOW_SIZE, 1])
        h_conv1 = tf.nn.relu(self.conv2d(input_2D, self.W_conv1) + self.b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2) + self.b_conv2)
        h_flat1 = tf.reshape(h_conv2, [-1, WINDOW_SIZE * WINDOW_SIZE * 10])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat1, self.W_fc1) + self.b_fc1)
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob) #add to feed_dict
        self.Qout = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2

        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.Qtarget = tf.placeholder(shape=[None, OUTPUT_SIZE], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Qtarget - self.Qout))
        self.trainStep = tf.train.AdamOptimizer(2*1e-3).minimize(self.loss)

        # initialize net:
        self.init = tf.global_variables_initializer()

        #   start session of game for learning:
        self.sess = tf.Session()
        self.sess.run(self.init)

        # data for supervised learning:
        self.rewardDict = {}
        self.lastObject = 0

        # data structures for chceck performance
        self.averageRewads = []
        self.allRewards = []

        #   set learning parameters:
        self.lamda = 0.8
        self.epsilon = 0.2 #will be deacrese during run

        # counter for learning time that was in game (start from -1 for list indexing), for debuging:
        self.learn_count = -1

    # functions for comfortable use of CNN:
    # initial random valus to the model:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # layer definition:
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # generic convolution and pooling:
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')




    def learn(self, reward, t):
        self.rewardDict[self.lastObject] = reward

        self.learn_count = self.learn_count +1
        self.allRewards.append(reward)

        if self.actions_count  < 0:
            return

        iter = self.actions_count % ARRAY_SIZE

        self.rewardsList[iter] = reward
        if iter > 1 and self.actions_count > IMITATION_ROUND_NUMBER:

            self.QtargetsList[iter-1][self.actionsList[iter-1]] = reward + self.lamda * np.max(self.QtargetsList[iter])
        if (iter %  BATCH_SIZE == 0) and (iter > 0):
            statesBatch = np.array(self.statesList[iter-BATCH_SIZE:iter])
            QtargetBatch = np.array(self.QtargetsList[iter-BATCH_SIZE:iter])

            _,l = self.sess.run([self.trainStep,self.loss],
                          feed_dict={self.inputs: statesBatch, self.Qtarget: QtargetBatch, self.keep_prob: 0.5})
            #
            # print the loss of the networks
            # if t % 100 ==0:
            #     print(np.mean(l))

        # dcrease the epsilon for random choices:
        if self.actions_count %1500 == 0 and  self.actions_count > 0 and self.epsilon > EPSILON_MINIMUM_VALUE:
            self.epsilon = self.epsilon * EPSILON_DECREASE_RATE

        # prints and plots for the avarages of the rewards:
        #
        # if  self.actions_count < 3000:
        #     if self.actions_count % 150 == 0 and self.actions_count > 0:
        #         self.averageRewads.append(sum(self.allRewards[-150:]) / 150)
        #
        # if(self.actions_count % 1000 == 0) and self.actions_count > 0:
        #     self.averageRewads.append(sum(self.allRewards[-1000:]) / 1000)
        #     print("action count: ", self.actions_count)
        #
        # if (self.actions_count % 10000 == 0) and self.actions_count > 0:
        #     print("last 9,000 mean:", sum(self.averageRewads[-9:])/9)
        #
        # if(self.actions_count%30000 == 0) and self.actions_count > 0:
        #     print("last 12,000 mean:", sum(self.averageRewads[-12:])/12)
        #     plt.plot(self.averageRewads)
        #     plt.show()


    # extracting window oround the snake head (WINDOW_SIZE*WINDOW_SIZE), and rotating it
    # to the direction of the snake
    def extractFitures(self, state, player_state):
        # return everything:
        dir = player_state['dir']
        rows = state.shape[0]
        cols = state.shape[1]
        rowWindowCorner = (player_state['chain'][-1][0] - int(WINDOW_SIZE / 2)) % rows
        colWindowCorner = (player_state['chain'][-1][1] - int(WINDOW_SIZE / 2)) % cols
        # fitures = np.zeros((WINDOW_SIZE * WINDOW_SIZE))
        features = np.zeros((WINDOW_SIZE, WINDOW_SIZE))
        for i in range(WINDOW_SIZE):
            for j in range(WINDOW_SIZE):
                rowCordinate = (rowWindowCorner + i) % rows
                colCordinate = (colWindowCorner + j) % cols
                # fitures[i * WINDOW_SIZE + j] = state[rowCordinate, colCordinate]
                features[i, j] = state[rowCordinate, colCordinate]
        if dir == 'E':
            features = np.rot90(features)
        elif dir == 'W':
            features = np.rot90(features, 3)
        elif dir == 'S':
            features = np.rot90(features, 2)

        # return the fetures as a flat vector
        features = features.flatten()
        return features


    # function for get feature of avoide collision (we used it for trying policies that we don't use in the end)
    def avoidCollision(self, state, player_state):
        avoidCollisionChoice = np.zeros(3)
        head_pos = player_state['chain'][-1]
        for a in list(np.random.permutation(bp.Policy.ACTIONS)):
            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][a]) % state.shape
            if state[r, c] <= 0:
                avoidCollisionChoice[ACTION_TO_INDEX[a]] = 1
            else:
                avoidCollisionChoice[ACTION_TO_INDEX[a]] = -1
        return avoidCollisionChoice

    #function for know which action go to best place by naive dictionary (use for imitation learning):
    def wiseAgent(self, state, player_state):
        ratedOptions = np.zeros(3)
        head_pos = player_state['chain'][-1]
        for a in list(np.random.permutation(bp.Policy.ACTIONS)):
            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][a]) % state.shape
            if state[r, c] in self.rewardDict:
                ratedOptions[ACTION_TO_INDEX[a]] = self.rewardDict[state[r, c]]
            else:
                ratedOptions[ACTION_TO_INDEX[a]] = 1000
        choice = np.argmax(ratedOptions)
        if ratedOptions[choice] == ratedOptions[2]: #for default go strate
            choice = 2
        #     remember to which object we coose to go from the state that we was (using integer representation)
        self.lastObject = state[
            head_pos.move(bp.Policy.TURNS[player_state['dir']][bp.Policy.ACTIONS[choice]]) % state.shape]
        return (choice, ratedOptions)

    # take action in game:
    def act(self, t, state, player_state):
        # count the number of actions that we take by ourselves
        self.actions_count = self.actions_count + 1

        # action index in the lists of the data (states, Q, etc.)
        action_idx = self.actions_count % ARRAY_SIZE

        # naive good choice for imitation learning:
        action, wiseChoice = self.wiseAgent(state, player_state)

        # choose next action using the net or randomly (epsilon greedily):
        features = self.extractFitures(state, player_state)
        # features = np.append(np.asarray(features), wiseChoice)
        self.statesList[action_idx] = features
        features = [features]
        stateQ = wiseChoice #in the first steps, we want to use the wise choice as a Q (actualy it's emetation learning)

        # the action will be take by the CNN
        if  self.actions_count > IMITATION_ROUND_NUMBER:
            action, stateQ = self.sess.run([self.predict, self.Qout], feed_dict={self.inputs:features, self.keep_prob: 1})
            action = action[0]
            stateQ = stateQ[0]

        # epsilon probability for random actions:
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(3)

        # remember curr action, and the stateQ:
        self.actionsList[action_idx] = action
        self.QtargetsList[action_idx] = stateQ #use the state's Q as a temp Q target (will be update in the learn function)

        # do action:
        return bp.Policy.ACTIONS[action]

    def get_state(self):
        # CNN variables:
        # first convolutional layer:
        W_conv1 = self.sess.run(self.W_conv1)
        b_conv1 = self.sess.run(self.b_conv1)

        # scond convolutional layer:
        W_conv2 = self.sess.run(self.W_conv2)
        b_conv2 = self.sess.run(self.b_conv2)

        # Densely Connected Layer (flat connection):
        W_fc1 = self.sess.run(self.W_fc1)
        b_fc1 = self.sess.run(self.b_fc1)

        #last conection to output:
        W_fc2 = self.sess.run(self.W_fc2)
        b_fc2 = self.sess.run(self.b_fc2)

        # remember the CNN variables, and lists that important for the imitation learning (the naive wise snake):
        self.state = (W_conv1,b_conv1,W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2,
                        self.actions_count, self.rewardDict, self.lastObject,self.statesList, self.rewardsList,
                        self.QtargetsList, self.actionsList)

        return self.state