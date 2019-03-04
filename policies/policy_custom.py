from policies import base_policy as bp
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

LEARNING_RATE = 0.3
GAMMA = 0.9
SHIFT = 1
EPSILON = 0.1
ROTATIONS = {'N':0, 'S':2, 'W':-1, 'E':1}
NUM_OF_FEATURES = 11

class Custom(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['rate'] = float(policy_args['rate']) if 'rate' in policy_args else LEARNING_RATE
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['shift'] = float(policy_args['shift']) if 'shift' in policy_args else SHIFT
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.model = Sequential()
        self.model.add(Dense(10, activation='relu', input_shape=(9*NUM_OF_FEATURES)))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(3))
        self.model.summary()
        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            # get window around the head
            small_board = self.get_small_board(prev_state)
            # get feature vector according the features in the window
            feature_vector = self.get_feature_vector(small_board)
            # get result for the last move
            result = self.model.predict(feature_vector)

            # get the next feature vector to calculate the next Q value
            next_small_board = self.get_small_board(new_state)
            next_feature_vector = self.get_feature_vector(next_small_board)

            # get the "label" as array of values for action, and the last action updates with the given reward
            result[bp.Policy.ACTIONS.index(prev_action)] -= reward+self.gamma*(np.max(self.model.predict(next_feature_vector)))
            self.model.fit(feature_vector, result, batch_size=1, nb_epoch=1, verbose=0)


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

    def get_small_board(self, new_state):
        """
        function to get small window around the head
        :param new_state: the current state (head, diretion and board)
        :return: window around the head / patch of the board around the head
        """
        board, head = new_state
        head_pos, direction = head
        rows, cols = board.shape
        new_head = list(head_pos)
        if head_pos[1]==0:
            board = np.roll(board,1,axis=1)
            new_head[1]+=1
            if head_pos[0]==0:
                board = np.roll(board,1,axis=0)
                new_head[0]+=1
            elif head_pos[0]==rows-1:
                board = np.roll(board,-1,axis=0)
                new_head[0]-=1
        elif head_pos[1]==cols-1:
            board = np.roll(board,-1,axis=1)
            new_head[1]-=1
            if head_pos[0]==0:
                board = np.roll(board,1,axis=0)
                new_head[0]+=1
            elif head_pos[0]==rows-1:
                board = np.roll(board,-1,axis=0)
                new_head[0]-=1
        return np.rot90(board[new_head[0]-1:new_head[0]+2,new_head[1]-1:new_head[1]+2], ROTATIONS[direction])

    def get_feature_vector(self, board):
        """
        get the feature vector
        :param board: the current given board
        :return: HOT vector that indicates the feature in the relevant index in the vector
        """
        feature_vector = np.zeros(board.size*NUM_OF_FEATURES)
        for i in range(board.size):
            row, col = i//board.shape[1],i%board.shape[1]
            feature_vector[i*NUM_OF_FEATURES+board[row,col]] = 1
        return feature_vector

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round%200==199:
            self.epsilon*=0.9
        # get window around the head
        small_board = self.get_small_board(new_state)
        # get feature vector according the features in the window
        feature_vector = self.get_feature_vector(small_board)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            print("RANDOM:", action)
            return action

        else:
            prediction = self.model.predict(feature_vector)
            action = bp.Policy.ACTIONS[np.argmax(prediction)]
            print("PREDICTION:", prediction, "| ACTION:", action)
            return action