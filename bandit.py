import sys
import numpy as np

num_arms = 10
num_steps = 10000
num_runs = 300
epsilon = 0.1
alpha = 0.1
initial_q = 0

def update_arms_noise(arm_values, n_arms):
    return arm_values + np.random.normal(0, .01, n_arms)

def epsilon_greedy_action(epsilon, q_values):
    if np.random.random() < epsilon:
        return np.random.choice(len(q_values))
    else:
        return np.argmax(q_values)
    
def update_q_values_sample_avg(reward, q_values_sample_avg, sample_avg_action, n_action_pulls):
    q_values_sample_avg[sample_avg_action] += (reward - q_values_sample_avg[sample_avg_action]) / n_action_pulls
    
def update_q_values_constant_step(reward, q_values_constant_step, constant_step_action, alpha):
    q_values_constant_step[constant_step_action] += alpha * (reward - q_values_constant_step[constant_step_action])


def bandit_sim():
    rewards_sample_avg = np.zeros((num_runs, num_steps))
    rewards_constant_step = np.zeros((num_runs, num_steps))
    optimal_action_sample_avg = np.zeros((num_runs, num_steps))
    optimal_action_constant_step = np.zeros((num_runs, num_steps))
    
    for run in range(num_runs):
        actual_arm_values = np.zeros(num_arms)
        q_values_sample_avg = np.zeros(num_arms) + initial_q
        q_values_constant_step = np.zeros(num_arms) + initial_q
        total_n_pulls = np.zeros(num_arms)
        
        for step in range(num_steps):
            actual_arm_values = update_arms_noise(actual_arm_values, num_arms)
            actual_optimal_action = np.argmax(actual_arm_values)
            
            # Sample Average
            sample_avg_action = epsilon_greedy_action(epsilon, q_values_sample_avg)
            reward = np.random.normal(actual_arm_values[sample_avg_action], 1)
            total_n_pulls[sample_avg_action] += 1
            update_q_values_sample_avg(reward, q_values_sample_avg, sample_avg_action, total_n_pulls[sample_avg_action])
            rewards_sample_avg[run, step] = reward
            optimal_action_sample_avg[run, step] = 1 if sample_avg_action == actual_optimal_action else 0
            
            # Constant Step
            constant_step_action = epsilon_greedy_action(epsilon, q_values_constant_step)
            reward = np.random.normal(actual_arm_values[constant_step_action], 1)
            update_q_values_constant_step(reward, q_values_constant_step, constant_step_action, alpha)
            rewards_constant_step[run, step] = reward
            optimal_action_constant_step[run, step] = 1 if constant_step_action == actual_optimal_action else 0
        
    # Averaging over all steps in each run
    avg_rewards_sample_avg = np.mean(rewards_sample_avg, axis=0)
    avg_rewards_constant_step = np.mean(rewards_constant_step, axis=0)
    avg_optimal_action_sample_avg = np.mean(optimal_action_sample_avg, axis=0)
    avg_optimal_action_constant_step = np.mean(optimal_action_constant_step, axis=0)
    
    return avg_rewards_sample_avg, avg_optimal_action_sample_avg, avg_rewards_constant_step, avg_optimal_action_constant_step
            
            
if __name__ == "__main__":
    avg_rewards_sample_avg, avg_optimal_action_sample_avg, avg_rewards_constant_step, avg_optimal_action_constant_step = bandit_sim()
    
    results_array = np.vstack((avg_rewards_sample_avg, avg_optimal_action_sample_avg, avg_rewards_constant_step, avg_optimal_action_constant_step))
    
    fname = sys.argv[1]
    np.savetxt(fname, results_array)