from policies import base_policy as bp
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import mnist

GAMMA = 0.8
EPSILON = 0.6
MIN_EPSILON = 0.1
EPSILON_RATE = 0.9
ROTATIONS = {'N':0, 'S':2, 'W':-1, 'E':1}
NUM_OF_FEATURES = 11
WINDOW_SIZE = 9
EPSILON_ROUNDS = 20
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 256
NN = 30
SHIFT = 1
class Custom200863793(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['min_epsilon'] = float(policy_args['min_epsilon']) if 'min_epsilon' in policy_args else MIN_EPSILON
        policy_args['epsilon_rate'] = float(policy_args['epsilon_rate']) if 'epsilon_rate' in policy_args else EPSILON_RATE
        policy_args['epsilon_rounds'] = float(policy_args['epsilon_rounds']) if 'epsilon_rounds' in policy_args else EPSILON_ROUNDS
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else MAX_BATCH_SIZE
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.model = Sequential()
        self.model.add(Dense(NN, activation='relu', input_shape=(WINDOW_SIZE*NUM_OF_FEATURES,)))
        self.model.add(Dense(NN, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae'])
        self.batch_counter = 0
        self.last_feature_vector = None
        self.batch_states = np.zeros((self.batch_size, WINDOW_SIZE*NUM_OF_FEATURES))
        self.next_step_feature_vectors = np.zeros((self.batch_size,3, WINDOW_SIZE*NUM_OF_FEATURES))
        self.rewards = np.zeros(self.batch_size)
        self.batches_to_take = MIN_BATCH_SIZE
        self.full_batch = False
        # self.last_mask = None
        # self.masks = np.zeros((self.batch_size,3, 3))
        # self.next_masks = np.zeros((self.batch_size,3, 3, 3))

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:

            # case we did not collect enough samples
            if not self.full_batch:
                self.batches_to_take = self.batch_counter
            if too_slow:
                self.batches_to_take = max(MIN_BATCH_SIZE, self.batches_to_take//2)

            sort_rewards = np.argsort(np.fabs(self.rewards)) # sort rewards because we want the best learning
            indices_to_take = sort_rewards[-self.batches_to_take:]

            # get the max Q values for the last "size of batch" examples
            Q_values = self.model.predict_on_batch(self.next_step_feature_vectors[indices_to_take].reshape((3*self.batches_to_take, NUM_OF_FEATURES*WINDOW_SIZE)))
            max_Q_values = np.reshape(Q_values, (self.batches_to_take,len(bp.Policy.ACTIONS))).max(axis=1)

            labels = self.rewards[indices_to_take] + (self.gamma*max_Q_values)

            self.model.train_on_batch(self.batch_states[indices_to_take], labels)

            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def get_vector(self, board):
        curr_feature_vector = np.zeros(WINDOW_SIZE*NUM_OF_FEATURES)
        for j in range(board.size):
            row, col = j//board.shape[1],j%board.shape[1]
            curr_feature_vector[j*NUM_OF_FEATURES+board[row,col]+1] = 1
        return curr_feature_vector

    def get_feature_vector(self, new_state):
        """
        return feature vectors according to action
        :param new_state:
        :return:
        """
        feature_vector = np.zeros((len(bp.Policy.ACTIONS),NUM_OF_FEATURES*WINDOW_SIZE))
        board, head = new_state
        head_pos, direction = head
        rows, cols = board.shape
        head_value = board[head_pos[0], head_pos[1]]
        # masks = []
        for i, action in enumerate(bp.Policy.ACTIONS):
            temp_direction = bp.Policy.TURNS[direction][action]
            new_head = list(head_pos.move(temp_direction))

            if new_head[1]==0:
                temp_board = np.roll(board,SHIFT,axis=1)
                new_head[1]+=SHIFT
                if new_head[0]==0:
                    temp_board = np.roll(temp_board,SHIFT,axis=0)
                    new_head[0]+=SHIFT
                elif new_head[0]==rows-1:
                    temp_board = np.roll(temp_board,-SHIFT,axis=0)
                    new_head[0]-=SHIFT
            elif new_head[1]==cols-1:
                temp_board = np.roll(board,-SHIFT,axis=1)
                new_head[1]-=SHIFT
                if new_head[0]==0:
                    temp_board = np.roll(temp_board,SHIFT,axis=0)
                    new_head[0]+=SHIFT
                elif new_head[0]==rows-1:
                    temp_board = np.roll(temp_board,-SHIFT,axis=0)
                    new_head[0]-=SHIFT

            else:
                if new_head[0]==0:
                    temp_board = np.roll(board,SHIFT,axis=0)
                    new_head[0]+=SHIFT
                elif new_head[0]==rows-1:
                    temp_board = np.roll(board,-SHIFT,axis=0)
                    new_head[0]-=SHIFT
                else:
                    temp_board = board.copy()
            crop_rotate_board = np.rot90(temp_board[new_head[0]-SHIFT:new_head[0]+SHIFT+1,new_head[1]-SHIFT:new_head[1]+SHIFT+1],ROTATIONS[temp_direction])
            curr_board = crop_rotate_board.copy()
            curr_board[1,1] = head_value
            # masks.append(curr_board)
            feature_vector[i,:] = self.get_vector(curr_board)

        # return feature_vector, masks
        return feature_vector

    def get_last_feature_vector(self, new_state, action):
        board, head = new_state
        head_pos, direction = head
        rows, cols = board.shape
        head_value = board[head_pos[0], head_pos[1]]
        temp_direction = bp.Policy.TURNS[direction][action]
        new_head = list(head_pos.move(temp_direction))
        if new_head[1]==0:
            temp_board = np.roll(board,SHIFT,axis=1)
            new_head[1]+=SHIFT
            if new_head[0]==0:
                temp_board = np.roll(temp_board,SHIFT,axis=0)
                new_head[0]+=SHIFT
            elif new_head[0]==rows-1:
                temp_board = np.roll(temp_board,-SHIFT,axis=0)
                new_head[0]-=SHIFT
        elif new_head[1]==cols-1:
            temp_board = np.roll(board,-SHIFT,axis=1)
            new_head[1]-=SHIFT
            if new_head[0]==0:
                temp_board = np.roll(temp_board,SHIFT,axis=0)
                new_head[0]+=SHIFT
            elif new_head[0]==rows-1:
                temp_board = np.roll(temp_board,-SHIFT,axis=0)
                new_head[0]-=SHIFT
        else:
            if new_head[0]==0:
                temp_board = np.roll(board,SHIFT,axis=0)
                new_head[0]+=SHIFT
            elif new_head[0]==rows-1:
                temp_board = np.roll(board,-SHIFT,axis=0)
                new_head[0]-=SHIFT
            else:
                temp_board = board.copy()
        crop_rotate_board = np.rot90(temp_board[new_head[0]-1:new_head[0]+2,new_head[1]-1:new_head[1]+2],ROTATIONS[temp_direction])
        crop_rotate_board[1,1] = head_value
        # return self.get_vector(crop_rotate_board), crop_rotate_board
        return self.get_vector(crop_rotate_board)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # epsilon decreases as we progress in the game
        if round%EPSILON_ROUNDS==EPSILON_ROUNDS-1 and self.epsilon>self.min_epsilon:
            self.epsilon*=self.epsilon_rate

        # get 3 feature vectors and their masks (window around the head)
        # feature_vector, masks = self.get_feature_vector(new_state)
        feature_vector= self.get_feature_vector(new_state)

        # case of bad or good performence change size of batch
        if too_slow:
            self.batches_to_take = max(MIN_BATCH_SIZE, self.batches_to_take//2)
        else:
            self.batches_to_take = min(MAX_BATCH_SIZE, self.batches_to_take+2)

        # preprocessing
        if prev_state and not too_slow:
            # self.masks[self.batch_counter] = self.last_mask     # save the last window
            # self.next_masks[self.batch_counter] = masks         # save the next windows
            self.batch_states[self.batch_counter] = self.last_feature_vector    # save the last feature vectors
            self.next_step_feature_vectors[self.batch_counter] = feature_vector # save the current feature vector
            self.rewards[self.batch_counter] = reward                           # save the last rewars
            self.batch_counter = (self.batch_counter + 1) % self.batch_size     # count how many states we save
            if not self.full_batch and self.batch_counter==self.batch_size-1:
                self.full_batch = True

        prediction = self.model.predict(feature_vector, batch_size=3)   # get predictions for the current optional states
        max_value = np.max(prediction)
        if prediction[2]==max_value:    # forward
            argmax = 2
        elif prediction[0]==prediction[1]:
            argmax = np.random.choice([0,1])    # random between right and left
        else:
            if prediction[0]==max_value:
                argmax = 0  # left
            else:
                argmax = 1  # right

        action = bp.Policy.ACTIONS[argmax]      # get the action according to the high prediction
        self.last_feature_vector = feature_vector[argmax]   # get the feature vector of this prediction
        # self.last_mask = masks[argmax]          # keep the mask of the high prediction

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            # keep the feature vector for the random action
            # self.last_feature_vector, self.last_mask = self.get_last_feature_vector(new_state, action)
            self.last_feature_vector = self.get_last_feature_vector(new_state, action)
            return action
        else:
            return action

