from policies import base_policy as bp
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import mnist

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
        self.model.add(Dense(10, activation='relu', input_shape=(9*NUM_OF_FEATURES,)))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(3))
        # self.model.summary()
        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # print("\n\nLEARN, reward:", self.epsilon ,"|slow:",too_slow)
        try:
            small_board = self.get_small_board(prev_state)
            feature_vector = self.get_feature_vector(small_board).reshape((1,99))
            result = self.model.predict(feature_vector)
            # print("last result:",result, "prev_action:", prev_action)

            # get the next feature vector to calculate the next Q value
            next_small_board = self.get_small_board(new_state)
            next_feature_vector = self.get_feature_vector(next_small_board).reshape((1,99))
            # print("next_small_board:\n",next_small_board)

            # get the "label" as array of values for action, and the last action updates with the given reward
            next_result = self.model.predict(next_feature_vector)
            # print("next_result:", next_result)

            result[0,bp.Policy.ACTIONS.index(prev_action)] = reward+self.gamma*(np.max(next_result))
            # print("update result:", result)

            # a = []
            # for layer in self.model.layers:
            #     a.append(layer.get_weights())
            #
            self.model.fit(feature_vector, result, verbose=0)
            # count = 0
            # for layer in self.model.layers:
            #     print(layer.get_weights()-a[count])
            #     count+=1


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
        else:
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
            feature_vector[i*NUM_OF_FEATURES+board[row,col]+1] = 1
        return feature_vector

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # print("act, epsilon:", self.epsilon)
        if round%20==19:
            self.epsilon*=0.9

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            # print("RANDOM:", action)
            return action

        else:
            # get window around the head
            small_board = self.get_small_board(new_state)
            # get feature vector according the features in the window
            # print(small_board)
            feature_vector = self.get_feature_vector(small_board).reshape((1,99))
            # print(feature_vector, "\n")
            prediction = self.model.predict(feature_vector)
            action = bp.Policy.ACTIONS[np.argmax(prediction)]
            # print("PREDICTION:", prediction, "| argmax:", np.argmax(prediction), "| ACTION:", action)
            return action

