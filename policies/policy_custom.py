from policies import base_policy as bp
import numpy as np
import keras
exit()
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import mnist

LEARNING_RATE = 0.3
GAMMA = 0.9
SHIFT = 1
EPSILON = 0.1
MIN_EPSILON = 0.01
EPSILON_RATE = 0.9
ROTATIONS = {'N':0, 'S':2, 'W':-1, 'E':1}
NUM_OF_FEATURES = 11
WINDOW_SIZE = 9
EPSILON_ROUNDS = 20
BATCH_SIZE = 5

class Custom(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['min_epsilon'] = float(policy_args['min_epsilon']) if 'min_epsilon' in policy_args else MIN_EPSILON
        policy_args['epsilon_rate'] = float(policy_args['epsilon_rate']) if 'epsilon_rate' in policy_args else EPSILON_RATE
        policy_args['epsilon_rounds'] = float(policy_args['epsilon_rounds']) if 'epsilon_rounds' in policy_args else EPSILON_ROUNDS
        policy_args['rate'] = float(policy_args['rate']) if 'rate' in policy_args else LEARNING_RATE
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['shift'] = float(policy_args['shift']) if 'shift' in policy_args else SHIFT
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.model = Sequential()
        self.model.add(Dense(10, activation='relu', input_shape=(WINDOW_SIZE*NUM_OF_FEATURES,)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae'])
        self.counter = 0
        self.last_feature_vector = None
        self.batch_states = np.zeros((self.batch_size, WINDOW_SIZE*NUM_OF_FEATURES))
        self.next_step_feature_vectors = np.zeros((self.batch_size,3, WINDOW_SIZE*NUM_OF_FEATURES))
        self.rewards = np.zeros(self.batch_size)
        self.full_batch = False
        self.last_mask = None
        self.mask = None

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # print("\n\nLEARN, reward:", self.epsilon ,"|slow:",too_slow)
        try:

            last_index = self.batch_size
            if not self.full_batch:
                last_index = self.counter

            max_Q_values = self.model.predict_on_batch(self.next_step_feature_vectors.reshape((3*self.batch_size, NUM_OF_FEATURES*WINDOW_SIZE))).max(axis=1)
            labels = self.rewards[:last_index] + (self.gamma*max_Q_values)
            # print("\nrewards:\n", self.rewards[:last_index])
            # print("max Q values:\n", self.max_Q_values[:last_index])
            # print("labels:\n", labels)
            # print("last states:\n", self.batch_states[:last_index])

            self.model.train_on_batch(self.batch_states[:last_index], labels)

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
        new_head = list(head_pos.move(direction))

        if head_pos[1]<=1:
            board = np.roll(board,2,axis=1)
            new_head[1]+=2
            if head_pos[0]<=1:
                board = np.roll(board,2,axis=0)
                new_head[0]+=2
            elif head_pos[0]>=rows-2:
                board = np.roll(board,-2,axis=0)
                new_head[0]-=2
        elif head_pos[1]>=cols-2:
            board = np.roll(board,-2,axis=1)
            new_head[1]-=2
            if head_pos[0]<=1:
                board = np.roll(board,2,axis=0)
                new_head[0]+=2
            elif head_pos[0]>=rows-2:
                board = np.roll(board,-2,axis=0)
                new_head[0]-=2
        else:
            if head_pos[0]<=1:
                board = np.roll(board,2,axis=0)
                new_head[0]+=2
            elif head_pos[0]>=rows-2:
                board = np.roll(board,-2,axis=0)
                new_head[0]-=2
        return np.rot90(board[new_head[0]-1:new_head[0]+2,new_head[1]-1:new_head[1]+2], ROTATIONS[direction])

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
        # head_value = board[head_pos[0], head_pos[1]]
        head_value = board[head_pos[0], head_pos[1]]
        # print("\n\nhead value:", head_value, "head_pos:", list(head_pos), "dir:", direction)
        masks = []
        for i, action in enumerate(bp.Policy.ACTIONS):
            temp_direction = bp.Policy.TURNS[direction][action]
            new_head = list(head_pos.move(temp_direction))
            # print("\nnew dir:", temp_direction, "new head_pos:", new_head)

            if new_head[1]==0:
                temp_board = np.roll(board,1,axis=1)
                new_head[1]+=1
                if new_head[0]==0:
                    temp_board = np.roll(temp_board,1,axis=0)
                    new_head[0]+=1
                elif new_head[0]==rows-1:
                    temp_board = np.roll(temp_board,-1,axis=0)
                    new_head[0]-=1
            elif new_head[1]==cols-1:
                temp_board = np.roll(board,-1,axis=1)
                new_head[1]-=1
                if new_head[0]==0:
                    temp_board = np.roll(temp_board,1,axis=0)
                    new_head[0]+=1
                elif new_head[0]==rows-1:
                    temp_board = np.roll(temp_board,-1,axis=0)
                    new_head[0]-=1

            else:
                if new_head[0]==0:
                    temp_board = np.roll(board,1,axis=0)
                    new_head[0]+=1
                elif new_head[0]==rows-1:
                    temp_board = np.roll(board,-1,axis=0)
                    new_head[0]-=1
                else:
                    temp_board = board.copy()
            crop_rotate_board = np.rot90(temp_board[new_head[0]-1:new_head[0]+2,new_head[1]-1:new_head[1]+2],ROTATIONS[temp_direction])
            # print("crop:\n", crop_rotate_board)
            curr_board = crop_rotate_board.copy()
            curr_board[1,1] = head_value
            # print("action:", action, "| curr board:\n", curr_board)
            masks.append(curr_board)
            feature_vector[i,:] = self.get_vector(curr_board)

        return feature_vector, masks

    def get_last_feature_vector(self, new_state, action):
        feature_vector = np.zeros(NUM_OF_FEATURES*WINDOW_SIZE)
        board, head = new_state
        head_pos, direction = head
        rows, cols = board.shape
        head_value = board[head_pos[0], head_pos[1]]
        temp_direction = bp.Policy.TURNS[direction][action]
        new_head = list(head_pos.move(temp_direction))
        if new_head[1]==0:
            temp_board = np.roll(board,1,axis=1)
            new_head[1]+=1
            if new_head[0]==0:
                temp_board = np.roll(temp_board,1,axis=0)
                new_head[0]+=1
            elif new_head[0]==rows-1:
                temp_board = np.roll(temp_board,-1,axis=0)
                new_head[0]-=1
        elif new_head[1]==cols-1:
            temp_board = np.roll(board,-1,axis=1)
            new_head[1]-=1
            if new_head[0]==0:
                temp_board = np.roll(temp_board,1,axis=0)
                new_head[0]+=1
            elif new_head[0]==rows-1:
                temp_board = np.roll(temp_board,-1,axis=0)
                new_head[0]-=1
        else:
            if new_head[0]==0:
                temp_board = np.roll(board,1,axis=0)
                new_head[0]+=1
            elif new_head[0]==rows-1:
                temp_board = np.roll(board,-1,axis=0)
                new_head[0]-=1
            else:
                temp_board = board.copy()
        crop_rotate_board = np.rot90(temp_board[new_head[0]-1:new_head[0]+2,new_head[1]-1:new_head[1]+2],ROTATIONS[temp_direction])
        crop_rotate_board[1,1] = head_value
        return self.get_vector(crop_rotate_board), crop_rotate_board

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round%EPSILON_ROUNDS==EPSILON_ROUNDS-1 and self.epsilon>self.min_epsilon:
            self.epsilon*=self.epsilon_rate

        feature_vector, self.mask = self.get_feature_vector(new_state)

        if prev_state and not too_slow:
            self.batch_states[self.counter] = self.last_feature_vector
            self.next_step_feature_vectors[self.counter] = feature_vector
            self.rewards[self.counter] = reward
            self.counter = (self.counter + 1) % self.batch_size
            print("counter:", self.counter, "\treward:", reward, "last action:", prev_action)
            print("last_mask:\n", self.last_mask)
            print("NEXT STATE : ")
            for i in range(len(bp.Policy.ACTIONS)):
                print("action:", bp.Policy.ACTIONS[i])
                print(self.mask[i])
            if self.counter == self.batch_size-1 and not self.full_batch:
                self.full_batch = True

        prediction = self.model.predict(feature_vector, batch_size=3)
        # print("predictions:", prediction.T)
        argmax = np.argmax(prediction)
        action = bp.Policy.ACTIONS[argmax]
        # print("\n\nround:", round, "predictions:\n", prediction, "\naction:", action)
        self.last_feature_vector = feature_vector[argmax]
        self.last_mask = self.mask[argmax]

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            self.last_feature_vector, self.last_mask = self.get_last_feature_vector(new_state, action)
            # print("\nRANDOM action:", action)
            # print("mask:\n", mask)
            return action
        else:
            # print("action:", action,"\nmask:\n", masks[argmax] )
            return action

