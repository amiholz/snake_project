from policies import base_policy as bp
import numpy as np

EPSILON = 0.5
LEARNING_RATE = 0.1
NUM_OF_FEATURES = 11

class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['rate'] = float(policy_args['rate']) if 'rate' in policy_args else LEARNING_RATE
        return policy_args

    def init_run(self):
        self.r_sum = 0

        self.theta = np.zeros(2*NUM_OF_FEATURES)
        # self.theta[7] = self.theta[8] = 10
        # self.theta[NUM_OF_FEATURES] = 2

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            # self.theta
            delta = self.theta*self.get_small_state(new_state, )
            # update = self.rate*

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


    def get_small_state(self, new_state, action):
        board, head = new_state
        head_pos, direction = head

        next_position = head_pos.move(bp.Policy.TURNS[direction][a])
        r = next_position[0]
        c = next_position[1]
        small_state = np.zeros(2*NUM_OF_FEATURES)
        if action in ['R', 'L']:
            small_state[board[r,c]+1] = 1
        else:
            small_state[NUM_OF_FEATURES + board[r,c]+1] = 1
        return small_state


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            q = []
            rand_action_list = list(np.random.permutation(bp.Policy.ACTIONS))
            for a in rand_action_list:
                # get a Position object of the position in the relevant direction from the head:
                small_state = self.get_small_state(new_state, a)
                print(new_state[1][1], a, self.theta.dot(small_state))
                q.append(self.theta.dot(small_state))
            return rand_action_list[np.argmax(q)]

            # # if all positions are bad:
            # return np.random.choice(bp.Policy.ACTIONS)

