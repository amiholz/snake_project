from policies import base_policy as bp
import numpy as np

EPSILON = 0.5
LEARNING_RATE = 0.3
GAMMA = 0.9
NUM_OF_FEATURES = 11

class Linear200863793(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['rate'] = float(policy_args['rate']) if 'rate' in policy_args else LEARNING_RATE
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.rewards = np.zeros(2*NUM_OF_FEATURES)
        self.theta = np.zeros(2*NUM_OF_FEATURES)

    def get_last_feature(self, prev_action, prev_state):
        board, head = prev_state
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][prev_action])
        feature = board[next_position[0],next_position[1]]
        if (prev_action in ['R', 'L']):
            return feature+1
        else:
            return feature+NUM_OF_FEATURES+1

    def get_best_value(self, new_state):
        q = []
        rand_action_list = list(np.random.permutation(bp.Policy.ACTIONS))
        for a in rand_action_list:
            weights = self.get_weights(new_state, a)
            q.append(self.theta.dot(weights))
        return np.max(q)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        #print("* learn *")
        if too_slow:
            print(too_slow)
        try:

            self.theta -= self.rate*self.rewards
            self.rewards = np.zeros(2*NUM_OF_FEATURES)

            #print("theta:\t"+str([ float('%.2f' % elem) for elem in self.theta[:11]]))
            #print("\t"+str([ float('%.2f' % elem) for elem in self.theta[11:]]))

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


    def get_weights(self, new_state, action):
        """
        :param new_state: board and head of current state
        :param action: the current action
        :return: weight vector with 1 in the relevant feature (action [L/R] and value)
        """
        board, head = new_state
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        r = next_position[0]
        c = next_position[1]
        weights = np.zeros(2*NUM_OF_FEATURES)
        if action in ['R', 'L']:
            weights[board[r,c]+1] = 1
        else:
            weights[NUM_OF_FEATURES + board[r,c]+1] = 1
        return weights

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        #print("epsilon:",self.epsilon)
        if round%200==199:
            self.epsilon*=0.9

        #print("theta:\t"+str([ float('%.2f' % elem) for elem in self.theta[:11]]))
        #print("\t"+str([ float('%.2f' % elem) for elem in self.theta[11:]]))

        if prev_state and not too_slow:
            feature = self.get_last_feature(prev_action, prev_state)
            weights = self.get_weights(prev_state, prev_action)

            #print("weights:\t"+str([ float('%.2f' % elem) for elem in weights[:11]]))
            #print("\t"+str([ float('%.2f' % elem) for elem in weights[11:]]))

            self.rewards[feature] += (self.theta.dot(weights) - (reward+self.gamma*self.get_best_value(new_state)))

            if feature<NUM_OF_FEATURES:
                self.rewards[feature] /=2
            #print("feature:", feature, "| reward:", reward, "| dot:%.2f"%self.theta.dot(weights), "| best:%.2f"% self.get_best_value(new_state))
            #print("rewards:\t"+str([ float('%.2f' % elem) for elem in self.rewards[:11]]))
            #print("\t\t"+str([ float('%.2f' % elem) for elem in self.rewards[11:]]))


        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            #print("*RANDOM*:\tdirection:",new_state[1][1], "| action:", action,"\n")
        else:
            q = []
            rand_action_list = list(np.random.permutation(bp.Policy.ACTIONS))
            for a in bp.Policy.ACTIONS:
                # get a Position object of the position in the relevant direction from the head:
                weights = self.get_weights(new_state, a)
                q.append(self.theta.dot(weights))
            #print("Q-value:",[ float('%.2f' % elem) for elem in q ])
            argmax = np.argmax(q)
            # I set the default direction is forward because turns is bad :(
            if argmax==2:
                action = bp.Policy.ACTIONS[2]
            elif argmax==1:
                if q[1]==q[2]:
                    action = bp.Policy.ACTIONS[2]
                else:
                    action = bp.Policy.ACTIONS[1]
            else:
                if q[0]>q[1]:
                    if q[0]>q[2]:
                        action = bp.Policy.ACTIONS[0]
                    else:
                        action = bp.Policy.ACTIONS[2]
                if q[0]==q[1]:
                    if q[1]==q[2]:
                        action = bp.Policy.ACTIONS[2]
                    else:
                        action = bp.Policy.ACTIONS[np.random.choice([0,1])]
            #print("direction:",new_state[1][1], "| action:", action,"\n")

        return action

