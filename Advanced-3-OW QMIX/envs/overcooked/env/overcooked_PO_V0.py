import gym
import numpy as np
from .render.game import Game
from gym import spaces
from .items import Tomato, Lettuce, Plate, Knife, Delivery, Agent, Food

import copy

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
TASKLIST = ["tomato salad", "lettuce salad", "tomato-lettuce salad"]

class POOvercooked_V0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
        }

    def __init__(self, grid_dim, task, rewardList, mapType = "A", debug = False):
        self.xlen, self.ylen = grid_dim
        if debug:
        	self.game = Game(self)

        self.task = task
        self.rewardList = rewardList
        self.mapType = mapType
        self.debug = debug

        map = []

        if self.xlen == 7 and self.ylen == 7:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 1],
                        [7, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 1],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 1],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]  
            elif self.mapType == "D":
                map =  [[1, 1, 1, 1, 3, 1, 1],
                        [6, 0, 2, 0, 2, 0, 1],
                        [6, 0, 0, 0, 0, 0, 4],
                        [7, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 5],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 5, 1, 1]]
            elif self.mapType == "E":
                map =  [[1, 1, 1, 1, 3, 1, 1],
                        [6, 0, 2, 1, 2, 0, 1],
                        [6, 0, 0, 1, 0, 0, 4],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 5],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 5, 1, 1]]
            elif self.mapType == "F":
                map =  [[1, 1, 1, 1, 3, 1, 1],
                        [6, 0, 2, 1, 2, 0, 1],
                        [6, 0, 0, 1, 0, 0, 4],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 5],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 5, 1, 1]]

        elif self.xlen == 9 and self.ylen == 9:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 0, 0, 1],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 1],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 1],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "D":
                map =  [[1, 1, 1, 1, 1, 1, 3, 1, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 1],
                        [6, 0, 0, 0, 0, 0, 0, 0, 4],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 5, 1, 1]]
            elif self.mapType == "E":
                map =  [[1, 1, 1, 1, 1, 1, 3, 1, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 1],
                        [6, 0, 0, 0, 1, 0, 0, 0, 4],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 5],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 5, 1, 1]]
            elif self.mapType == "F":
                map =  [[1, 1, 1, 1, 1, 1, 3, 1, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 1],
                        [6, 0, 0, 0, 1, 0, 0, 0, 4],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 5],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 5, 1, 1]]

        self.initMap = map
        self.map = copy.deepcopy(self.initMap)

        self.oneHotTask = []
        for t in TASKLIST:
            if t == self.task:
                self.oneHotTask.append(1)
            else:
                self.oneHotTask.append(0)

        self._createItems()
        self.n_agent = len(self.agent)

        #action: move(up, down, left, right), stay
        self.action_space = spaces.Discrete(5)

        #Observation: agent(pos[x,y]) dim = 2
        #    knife(pos[x,y]) dim = 2
        #    delivery (pos[x,y]) dim = 2
        #    plate(pos[x,y]) dim = 2
        #    food(pos[x,y]/status) dim = 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self._initObs()[0]),), dtype=np.float32)

    def _createItems(self):
        self.agent = []
        self.knife = []
        self.delivery = []
        self.tomato = []
        self.lettuce = []
        self.plate = []
        self.itemList = []
        agent_idx = 0
        for x in range(self.xlen):
            for y in range(self.ylen):
                if self.map[x][y] == ITEMIDX["agent"]:
                    self.agent.append(Agent(x, y, color = AGENTCOLOR[agent_idx]))
                    agent_idx += 1
                elif self.map[x][y] == ITEMIDX["knife"]:
                    self.knife.append(Knife(x, y))
                elif self.map[x][y] == ITEMIDX["delivery"]:
                    self.delivery.append(Delivery(x, y))                    
                elif self.map[x][y] == ITEMIDX["tomato"]:
                    self.tomato.append(Tomato(x, y))
                elif self.map[x][y] == ITEMIDX["lettuce"]:
                    self.lettuce.append(Lettuce(x, y))
                elif self.map[x][y] == ITEMIDX["plate"]:
                    self.plate.append(Plate(x, y))
        
        self.itemDic = {"tomato": self.tomato, "lettuce": self.lettuce, "plate": self.plate, "knife": self.knife, "delivery": self.delivery, "agent": self.agent,}
        for key in self.itemDic:
            self.itemList += self.itemDic[key]

    def _initObs(self):
        obs = []
        for item in self.itemList:
            obs.append(item.x / self.xlen)
            obs.append(item.y / self.ylen)
            if isinstance(item, Food):
                obs.append(item.cur_chopped_times / item.required_chopped_times)
        obs += self.oneHotTask 

        for agent in self.agent:
            agent.obs = obs
        return [np.array(obs)] * self.n_agent

    def _getObs(self):
        po_obs = []

        for agent in self.agent:
            obs = []
            idx = 0
            if self.xlen == 7 and self.ylen == 7:
                if self.mapType == "A" or self.mapType == "D":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B" or self.mapType == "E":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C" or self.mapType == "F":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
            elif self.xlen == 9 and self.ylen == 9:
                if self.mapType == "A" or self.mapType == "D":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B" or self.mapType == "E":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C" or self.mapType == "F":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]

            for item in self.itemList:
                if item.x >= agent.x - 2 and item.x <= agent.x + 2 and item.y >= agent.y - 2 and item.y <= agent.y + 2:
                    x = item.x / self.xlen
                    y = item.y / self.ylen
                    obs.append(x)
                    obs.append(y)
                    idx += 2
                    if isinstance(item, Food):
                        obs.append(item.cur_chopped_times / item.required_chopped_times)
                        idx += 1
                else:
                    x = agent.obs[idx] * self.xlen
                    y = agent.obs[idx + 1] * self.ylen
                    if x >= agent.x - 2 and x <= agent.x + 2 and y >= agent.y - 2 and y <= agent.y + 2:
                        x = item.initial_x
                        y = item.initial_y
                    x = x / self.xlen
                    y = y / self.ylen

                    obs.append(x)
                    obs.append(y)
                    idx += 2
                    if isinstance(item, Food):
                        obs.append(agent.obs[idx] / item.required_chopped_times)
                        idx += 1

                agent.pomap[int(x * self.xlen)][int(y * self.ylen)] = ITEMIDX[item.rawName]
            agent.pomap[agent.x][agent.y] = ITEMIDX["agent"]
            obs += self.oneHotTask 
            agent.obs = obs
            po_obs.append(np.array(obs))

        return po_obs
    
    def _findItem(self, x, y, itemName):
        for item in self.itemDic[itemName]:
            if item.x == x and item.y == y:
                return item
        return None

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    @property
    def action_spaces(self):
        return [self.action_space] * self.n_agent

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    
    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    def reset(self):
        self.map = copy.deepcopy(self.initMap)
        self._createItems()
        self._initObs()
        if self.debug:
            self.game.on_cleanup()

        return self._getObs()
    
    def step(self, action):
        reward = self.rewardList["step penalty"]
        done = False
        info = {}
        info['cur_mac'] = action
        info['mac_done'] = [True] * self.n_agent
        info['collision'] = []

        all_action_done = False

        for agent in self.agent:
            agent.moved = False

        if self.debug:
            print("in overcooked primitive actions:", action)

        if action[0] < 4 and action[1] < 4:
            new_agent0_x = self.agent[0].x + DIRECTION[action[0]][0]
            new_agent0_y =  self.agent[0].y + DIRECTION[action[0]][1] 

            new_agent1_x =  self.agent[1].x + DIRECTION[action[1]][0]
            new_agent1_y =  self.agent[1].y + DIRECTION[action[1]][1]    

            if new_agent0_x == self.agent[1].x and new_agent0_y == self.agent[1].y\
            and new_agent1_x == self.agent[0].x and new_agent1_y == self.agent[0].y:
                self.agent[0].move(new_agent0_x, new_agent0_y)
                self.agent[1].move(new_agent1_x, new_agent1_y)
                if self.debug:
                    print("swap")
                return self._getObs(), [reward] * self.n_agent, done, info

        while not all_action_done:
            for idx, agent in enumerate(self.agent):
                agent_action = action[idx]
                if agent.moved:
                    continue
                agent.moved = True

                if agent_action < 4:
                    target_x = agent.x + DIRECTION[agent_action][0]
                    target_y = agent.y + DIRECTION[agent_action][1]
                    target_name = ITEMNAME[self.map[target_x][target_y]]

                    if target_name == "agent":
                        target_agent = self._findItem(target_x, target_y, target_name)
                        if not target_agent.moved:
                            agent.moved = False
                        else:
                            info['collision'].append(idx)
                            if self.debug:
                                print("collision !", info['collision'])
                    elif  target_name == "space":
                        self.map[agent.x][agent.y] = ITEMIDX["space"]
                        agent.move(target_x, target_y)
                        self.map[target_x][target_y] = ITEMIDX["agent"]
                    #pickup and chop
                    elif not agent.holding:
                        if target_name == "tomato" or target_name == "lettuce" or target_name == "plate":
                            item = self._findItem(target_x, target_y, target_name)
                            agent.pickup(item)
                            self.map[target_x][target_y] = ITEMIDX["counter"]
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            if isinstance(knife.holding, Plate):
                                item = knife.holding
                                knife.release()
                                agent.pickup(item)
                            elif isinstance(knife.holding, Food):
                                if knife.holding.chopped:
                                    item = knife.holding
                                    knife.release()
                                    agent.pickup(item)
                                else:
                                    knife.holding.chop()
                                    if knife.holding.chopped:
                                        if self.task == "tomato-lettuce salad"\
                                        or (self.task == "tomato salad" and isinstance(knife.holding, Tomato)) \
                                        or (self.task == "lettuce salad" and isinstance(knife.holding, Lettuce)):
                                            reward += self.rewardList["subtask finished"]
                    #put down
                    elif agent.holding:
                        if target_name == "counter":
                            if isinstance(agent.holding, Tomato):
                                self.map[target_x][target_y] = ITEMIDX["tomato"]
                            elif isinstance(agent.holding, Lettuce):
                                self.map[target_x][target_y] = ITEMIDX["lettuce"]
                            elif isinstance(agent.holding, Plate):
                                self.map[target_x][target_y] = ITEMIDX["plate"]
                            agent.putdown(target_x, target_y)
                        elif target_name == "plate":
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    plate = self._findItem(target_x, target_y, target_name)
                                    item = agent.holding
                                    agent.putdown(target_x, target_y)
                                    plate.contain(item)
                                    
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            if not knife.holding:
                                item = agent.holding
                                agent.putdown(target_x, target_y)
                                knife.hold(item)
                            elif isinstance(knife.holding, Food) and isinstance(agent.holding, Plate):
                                item = knife.holding
                                if item.chopped:
                                    knife.release()
                                    agent.holding.contain(item)
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Food):
                                plate_item = knife.holding
                                food_item = agent.holding
                                if food_item.chopped:
                                    knife.release()
                                    agent.pickup(plate_item)
                                    agent.holding.contain(food_item)
                        elif target_name == "delivery":
                            if isinstance(agent.holding, Plate):
                                if agent.holding.containing:
                                    if len(agent.holding.containing) > 1 and self.task == "tomato-lettuce salad":
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        self.delivery[0].hold(item)
                                        reward += self.rewardList["correct delivery"]
                                        done = True
                                    elif len(agent.holding.containing) == 1 and \
                                        ((agent.holding.containing[0].rawName == "tomato" and self.task == "tomato salad") \
                                        or (agent.holding.containing[0].rawName == "lettuce" and self.task == "lettuce salad")):
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        self.delivery[0].hold(item)
                                        reward += self.rewardList["correct delivery"]
                                        done = True
                                    else:
                                        reward += self.rewardList["wrong delivery"]
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        food = item.containing
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        for f in food:
                                            f.refresh()
                                            self.map[f.x][f.y] = ITEMIDX[f.rawName]
                                else:
                                    reward += self.rewardList["wrong delivery"]
                                    plate = agent.holding
                                    agent.putdown(target_x, target_y)
                                    plate.refresh()
                                    self.map[plate.x][plate.y] = ITEMIDX[plate.name]
                            else:
                                reward += self.rewardList["wrong delivery"]
                                food = agent.holding
                                agent.putdown(target_x, target_y)
                                food.refresh()
                                self.map[food.x][food.y] = ITEMIDX[food.rawName]

                        elif target_name == "tomato" or target_name == "lettuce":
                            item = self._findItem(target_x, target_y, target_name)
                            if item.chopped and isinstance(agent.holding, Plate):
                                agent.holding.contain(item)
                                self.map[target_x][target_y] = ITEMIDX["counter"]

            all_action_done = True
            for agent in self.agent:
                if agent.moved == False:
                    all_action_done = False
        
        return self._getObs(), [reward] * self.n_agent, done, info

    def render(self, mode='human'):
        return self.game.on_render()

    





