from policies import base_policy as bp
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

LEARNING_RATE = 0.3
GAMMA = 0.9
SHIFT = 1
EPSILON = 0.1

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
        # self.model = Sequential()
        # self.model.add(Convolution2D(5, 3, 3, activation='relu', input_shape=(1,3,3)))
        # self.model.add(Convolution2D(5, 3, 3, activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2,2)))
        # self.model.add(Dropout(0.25))
        # self.model.add(Flatten())
        # self.model.add(Dense(3, activation='softmax'))
        # self.model.compile(loss='mean_squared_error',
        #               optimizer='adam',
        #               metrics=['accuracy'])


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:

            # self.model.fit(X_train, Y_train,
            #           batch_size=1, nb_epoch=10, verbose=1)



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

    def get_small_board(self, new_state, shift):
        board, head = new_state
        head_pos, direction = head
        rows, cols = board.shape

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        small_board = self.get_small_board(new_state, self.shift)

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:

            # action = np.argmax(self.model.predict())
            return np.random.choice(bp.Policy.ACTIONS)