import numpy as np


#------------Define parameters---------------------
t = 5 #horizon
states = ["Tiger left","Tier right"] #states
actions = ["Open left","Open right","Listen"] #actions
observations = ["observe tiger left","observe tiger right"] #observations
belief_initial = [0.5,0.5]
gamma = 0.85

#Transition probabilities [state_next][state][action]
T =  np.array([[[0.5,0.5,1],
               [0.5,0.5,0]],
              [[0.5,0.5,0],
              [0.5,0.5,1]],
              ])

#-------Define Observation probabilities-----------
#Observation probabilities [observation][state][action]
O = np.array([[[0.5, 0.5, 0.85],
               [0.5,0.5, 0.15]],
              [[0.5,0.5, 0.15],
              [0.5,0.5, 0.85]],
             ])

#-----------------Reward setting--------------------
R = np.array([[-100,10,-1],[10,-100,-1]])



class tiger_problem:
    def  __init__(self, horizon, states,actions,observations,belief_intial,gamma,tran_matrix,obs_matrix,R_func):
        self.horizon = horizon
        self.states = states
        self.actions = actions
        self.observations = observations
        self.b_init = belief_initial
        self.tran_matrix = tran_matrix
        self.obs_matrix = obs_matrix
        self.R_func = R_func
        self.gamma = gamma

    # ------------belief update-------------------------
    def belief_update(self, action, obs, b):
        b_new = [0,0]
        for sj in range(len(states)):
            pr_obs = self.obs_matrix[obs][sj][action]
            summation = 0.0
            for si in range(len(states)):
                pr_s_prime = self.tran_matrix[sj][si][action]
                summation += pr_s_prime * b[si]
            b_new[sj] = (pr_obs * summation)

        # --- normalize the whole probability----
        total = sum(b_new)
        b_new = [x / total for x in b_new]
        print("new belief",b_new)
        # print("new belief",b_new)
        return b_new

    #------------Value iteration-----------------------
    def value(self,b,k):
        # return value_future
        if k == self.horizon:
            return 0
        if (k < self.horizon):
            Vmax = -10000
            rw_exp_vec = np.matmul(b,R)
            # print("--------------")
            # print("rw_exp_vec",rw_exp_vec)
            for a in range(len(actions)):
                value_future = 0
                print("a is ",a)
                rw_exp = rw_exp_vec[a]
                print("rw_exp_s is ", rw_exp)
                for obs in range(len(observations)):
                    b_new = self.belief_update(a, obs, b)
                    v_b_new=self.value(b_new, k+1)
                    for sj in range(len(states)):
                        for si in range(len(states)):
                            # print(k)
                            value_future += b[si]*self.tran_matrix[sj][si][a]*self.obs_matrix[obs][si][a]*v_b_new
                            #print("value future is ",value_future)
                Value = (rw_exp + gamma*value_future)
                print("***")
                print("depth:", k, " action:", actions[a])
                print("value ", Value, " reward", rw_exp, " future", value_future)

                if Value > Vmax:
                    #print("V_max is ", Value)
                    a_max = a
                    #print("action optimal",a_max)
                    Vmax = Value

        print("----")
        print("depth:", k, " action:", actions[a])
        print("V_max is ", Vmax)
        print("action optimal", actions[a_max])
        return Vmax



test = tiger_problem(t,states,actions,observations,belief_initial,gamma,T,O,R)
print(test.value(belief_initial,0))
