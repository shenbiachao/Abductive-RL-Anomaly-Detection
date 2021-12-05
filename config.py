from common import try_gpu

# common
eps = 1e-9

# main

# loader
contamination_rate = 0.04
anomaly_num = 60
train_percentage = 0.8
labeled_percentage = 0.1
device = try_gpu(0)

# trainer
batch_size = 32
max_buffer_size = 100000
steps_per_iteration = 2000
max_iteration = 100
init_epsilon = 1
final_epsilon = 0.1
test_interval = 10
num_test_trajectories = 5
start_timestep = 1000
action_noise_scale = 0.1
log_interval = 100


# agent
gamma = 0.99
rl_learning_rate = 0.002
agent_hidden_dims = 256
target_smoothing_tau = 0.5
update_target_network_interval = 500
upper_bound = 1
lower_bound = -1

# classifier
classifier_learning_rate = 0.01
sample_size = 500
anomaly_class_num = 2
hidden_dim = 5
refresh_num = 10
