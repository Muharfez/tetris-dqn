from collections import deque
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import tetris as env 

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.95
UPDATE_TARGET = 5
MODEL_NAME = '256*2'
EPISODES = 20_000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

class DQNAgent:
    def __init__(self) -> None:
        self.policy_model = self.create_model()
        
        self.target_model = self.create_model()
        self.target_model.set_weights(self.policy_model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256,3,input_shape=(20,10,1),))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,3,))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(1,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(learning_rate=0.001),metrics=['accracy'])

        return model

    def update_replay_memory(self,transiiton):
        self.replay_memory.append(transiiton)

    def get_q(self,state):
        return self.policy_model.predict(np.array(state).reshape(-1,*state.shape))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory ,MINIBATCH_SIZE)
        
        states = np.array([transition[0] for transition in minibatch])
        grids = np.array([state[0] for state in states])
        
        next_pieces = np.array([state[1] for state in states])
        X = []
        Y = [] 
        for index, (state, reward, done) in enumerate(minibatch):
            if not done: 
                max_future_q = -1000
                possible_grids = env.get_all_possible_moves(grids[index],next_pieces[index])
                possible_grids = np.array([possible_grid[0] for possible_grid in possible_grids])
                for possible_grid in possible_grids:
                    temp = (self.target_model.predict(np.array([possible_grid]))[0])
                    if temp > max_future_q:
                        max_future_q = temp
                target_q = reward + DISCOUNT * max_future_q
            else:
                target_q = reward

            new_reward = target_q

            #adjust state
            X.append(state)
            Y.append(new_reward)

        self.policy_model.fit(np.array(X), np.array(Y), batch_size = MINIBATCH_SIZE,
            shuffle = False,verbose = 0,callbacks = [self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter +=1
        
        if self.target_update_counter > UPDATE_TARGET:
            self.target_model.set_weights(self.policy_model.get_weights())
            self.target_update_counter = 0




class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()

agent = DQNAgent()
for episode in tqdm(range(1,EPISODES + 1),ascii=True,unit='episode'):
    grid = env.reset()
    done = False
    while not done:
        current_piece = env.get_shape()
        possible_moves = env.get_all_possible_moves(grid,current_piece)
        if np.random.random() > epsilon:
            possible_grids = np.array([possible_move[0] for possible_move in possible_moves])
            index = 0
            for i,possible_grid in enumerate(possible_grids):
                temp = agent.get_q(possible_grid)
                if temp > max_future_q:
                    index = i
                    max_future_q = temp
            action = i
        else:
            action = np.random.randint(0,len(possible_moves))
        
        new_grid, reward, done = env.make_move(possible_moves,action)
        agent.update_replay_memory((new_grid,reward,done))
        agent.train(done)

        grid = env.copy_grid(new_grid)
        
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)