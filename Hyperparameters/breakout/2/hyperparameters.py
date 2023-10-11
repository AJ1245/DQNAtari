GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=1000000
MIN_REPLAY_SIZE=50000
EPSILON_START=1.0
EPSILON_END=0.1
NUM_ENVS = 4
EPSILON_DECAY=int(1e6)
TARGET_UPDATE_FREQ = 10000
LR = 0.00025
SAVE_PATH = './atari_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/atari_dqn'
LOG_INTERVAL = 100