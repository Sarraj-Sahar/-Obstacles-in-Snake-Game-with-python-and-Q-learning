import torch
import random
import numpy as np
from collections import deque
from game2 import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

#these params can be changed and expiremented with
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005  #from 0 to 1 , the higher the value --> the quicker the learning

class Agent:

    def __init__(self):
        self.n_games = 0 #nbr of games , initialized at 0
        self.epsilon = 0 # param to control randomness --> the higher the epsilon --> the higher the chosen q value --> less exploration
        self.gamma = 0.9 # discount rate / future reward value
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if we exceed the max memory
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0] # snake is a list and head is the first element

        #creating four points around the head to check for danger...
        point_l = Point(head.x - 20, head.y) #20 = blocksize
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # creating an array containing all the danger states
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or #if we're going right and the point to our right is a collision --> that point is a danger
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) #from bool to int values


    # we want to remember all of the above in a memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    # this is what we need to train for MULTIPLE (array of) game_steps
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    # this is all we need to train for ONE game_step
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration (at the begining)/ exploitation (for a better learning model)

        self.epsilon = 80 - self.n_games # a.the more games we have --> the smaller the epsilon gets
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: #b.the smaller the epsilon gets --> the less random action we take
            move = random.randint(0, 2)
            final_move[move] = 1

        # we get a move from our model :  a prediction
        else:
            state0 = torch.tensor(state, dtype=torch.float) #model
            prediction = self.model(state0) #based on state0
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = [] #start with empty list for scores ,will be used for plotting
    plot_mean_scores = [] #keep track of the mean score
    total_score = 0
    record = 0
    agent = Agent() #create an agent
    game = SnakeGameAI() #create a game

    while True: #run forever until we quit
        # get old state
        state_old = agent.get_state(game)

        # get move based on the current (old) state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move) #we call game.play_step(move) from game.py
        state_new = agent.get_state(game)

        # train short memory :for one action (the latest action)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1 #increase nbr of games
            agent.train_long_memory()

            if score > record: # check if we have a new highscore
                record = score
                agent.model.save()

            #this is what is being shown on the console
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            #plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()