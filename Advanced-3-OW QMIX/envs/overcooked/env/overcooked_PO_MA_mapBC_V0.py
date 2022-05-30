from PIL.Image import new
import gym
from gym.core import ActionWrapper
import numpy as np
from queue import PriorityQueue


from gym import spaces
from numpy.core.fromnumeric import _squeeze_dispatcher
from .items import Tomato, Lettuce, Plate, Knife, Delivery, Agent, Food
from .render.game import Game
from .overcooked_PO_V0 import POOvercooked_V0
from .mac_agent import MacAgent
from gym import Wrapper
import random



DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
TASKLIST = ["tomato salad", "lettuce salad", "tomato-lettuce salad"]
MACROACTIONNAME = ["stay", "get tomato", "get lettuce", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter", "right", "down", "left", "up"]
ACTIONIDX = {"right": 0, "down": 1, "left": 2, "up": 3, "stay": 4}
PRIMITIVEACTION =["right", "down", "left", "up", "stay"]

#macro action space
#get tomato: 0
#get lettuce: 1
#get plate 1/2: 2 3
#go to knife 1/2: 4 5
#chop: 6
#deliver: 7
#chop: 8
#go to counter: 9
#right: 10
#down: 11
#left: 12
#up: 13

class AStarAgent(object):
    def __init__(self, x, y, g, dis, action, history_action, pass_agent):
        self.x = x
        self.y = y
        self.g = g
        self.dis = dis
        self.action = action
        self.history_action = history_action
        self.pass_agent = pass_agent

    def __lt__(self, other):
        if self.dis != other.dis:
            return self.dis <= other.dis
        else:
            return self.pass_agent <= other.pass_agent


class POOvercooked_MA_mapBC_V0(POOvercooked_V0):
        
    def __init__(self, grid_dim, task, rewardList, mapType, debug):
        super().__init__(grid_dim, task, rewardList, mapType, debug)
        self.macroAgent = []
        self._createMacroAgents()
        self.macroActionItemList = []
        self._createMacroActionItemList()
        self.action_space = spaces.Discrete(len(MACROACTIONNAME))

        if self.xlen == 7 and self.ylen == 7:
            if self.mapType == "B" or self.mapType == "E":
                self.counterSequence = [3, 2, 4, 1]
            elif self.mapType == "C" or self.mapType == "F":
                self.counterSequence = [3, 2, 4, 1, 5]
        elif self.xlen == 9 and self.ylen == 9:
            if self.mapType == "B" or self.mapType == "E":
                self.counterSequence = [4, 3, 5, 2, 6, 1]
            elif self.mapType == "C" or self.mapType == "F":
                self.counterSequence = [4, 3, 5, 2, 6, 1, 7]


    def _createMacroAgents(self):
        for agent in self.agent:
            self.macroAgent.append(MacAgent())

    def _createMacroActionItemList(self):
        self.macroActionItemList = []
        for key in self.itemDic:
            if key != "agent":
                self.macroActionItemList += self.itemDic[key]

    def macro_action_sample(self):
        macro_actions = []
        for agent in self.agent:
            macro_actions.append(random.randint(0, self.action_space.n - 1))
        return macro_actions     

    def build_agents(self):
        raise

    def build_macro_actions(self):
        raise

    def _findPOitem(self, agent, macro_action):
        if macro_action < 3:
            idx = (macro_action - 1) * 3
        else:
            idx = (macro_action - 1) * 2 + 2
        
        return int(agent.obs[idx] * self.xlen), int(agent.obs[idx + 1] * self.ylen)

    def reset(self):
        super().reset()
        for agent in self.macroAgent:
            agent.reset()
        return self._get_macro_obs()

    def run(self, macro_actions):
        actions = self._computeLowLevelActions(macro_actions)
        
        obs, rewards, terminate, info = self.step(actions)

        self._checkMacroActionDone()
        self._checkCollision(info)
        cur_mac = self._collectCurMacroActions()
        mac_done = self._computeMacroActionDone()

        if self.debug:
            print("cur_mac name", MACROACTIONNAME[cur_mac[0]], MACROACTIONNAME[cur_mac[1]])
            print("cur_mac", cur_mac)
            print("mac_done", mac_done)
            print("primitive action:", PRIMITIVEACTION[actions[0]], PRIMITIVEACTION[actions[1]])

            print("#############################################")
            print("Blue Agent Action: ", MACROACTIONNAME[info['cur_mac'][0]])
            print("Action Done: ", info['mac_done'][0])
            print("Blue Agent Observation")
            print("tomato pos: ", obs[0][0:2]*7)
            print("tomato status: ", obs[0][2])
            print("lettuce pos: ", obs[0][3:5]*7)
            print("lettuce status: ", obs[0][5])
            print("plate-1 pos: ", obs[0][6:8]*7)
            print("plate-2 pos: ", obs[0][8:10]*7)
            print("knife-1 pos: ", obs[0][10:12]*7)
            print("knife-2 pos: ", obs[0][12:14]*7)
            print("delivery: ", obs[0][14:16]*7)
            print("agent-1: ", obs[0][16:18]*7)
            print("agent-2: ", obs[0][18:20]*7)
            print("order: ", obs[0][20:])
            print("#############################################")
            print("#############################################")
            print("Pink Agent Action: ", MACROACTIONNAME[info['cur_mac'][1]])
            print("Action Done: ", info['mac_done'][1])
            print("Pink Agent Observation")
            print("tomato pos: ", obs[1][0:2]*7)
            print("tomato status: ", obs[1][2])
            print("lettuce pos: ", obs[1][3:5]*7)
            print("lettuce status: ", obs[1][5])
            print("plate-1 pos: ", obs[1][6:8]*7)
            print("plate-2 pos: ", obs[1][8:10]*7)
            print("knife-1 pos: ", obs[1][10:12]*7)
            print("knife-2 pos: ", obs[1][12:14]*7)
            print("delivery: ", obs[1][14:16]*7)
            print("agent-1: ", obs[1][16:18]*7)
            print("agent-2: ", obs[1][18:20]*7)
            print("order: ", obs[1][20:])
            print("#############################################")

        self._createMacroActionItemList()

        info = {'cur_mac': cur_mac, 'mac_done': mac_done}
        return  self._get_macro_obs(), rewards, terminate, info

    def _checkCollision(self, info):
        for idx in info["collision"]:
            self.macroAgent[idx].cur_macro_action_done = True

    def _checkMacroActionDone(self):
        # loop each agent
        for idx, agent in enumerate(self.agent):
            if not self.macroAgent[idx].cur_macro_action_done:
                macro_action = self.macroAgent[idx].cur_macro_action
                if (MACROACTIONNAME[macro_action] == "go to knife 1"\
                    or MACROACTIONNAME[macro_action] == "go to knife 2") and not agent.holding:
                    target_x, target_y = self._findPOitem(agent, macro_action)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                elif (MACROACTIONNAME[macro_action] == "get tomato"\
                    or MACROACTIONNAME[macro_action] == "get lettuce"):
                    #when the food on the knife is not chopped, terminate
                    target_x, target_y = self._findPOitem(agent, macro_action)

                    macroAction2ItemName = {"get tomato": "tomato", "get lettuce": "lettuce"}
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        for knife in self.knife:
                            if knife.x == target_x and knife.y == target_y:
                                food = self._findItem(target_x, target_y, macroAction2ItemName[MACROACTIONNAME[macro_action]])
                                if not food.chopped:
                                    self.macroAgent[idx].cur_macro_action_done = True
                                    break
                elif MACROACTIONNAME[macro_action] == "deliver" and not agent.holding:
                    target_x, target_y = self._findPOitem(agent, macro_action)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                elif MACROACTIONNAME[macro_action] == "go to counter " and not agent.holding:
                    target_x = 0
                    target_y = int(self.ylen // 2)
                    findEmptyCounter = False
                    for i in self.counterSequence:
                        if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                            target_x = i
                            findEmptyCounter = True
                            break
                    if findEmptyCounter:
                        if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                            self.macroAgent[idx].cur_macro_action_done = True
                    else:
                        self.macroAgent[idx].cur_macro_action_done = True

                if MACROACTIONNAME[macro_action] == "get tomato" or MACROACTIONNAME[macro_action] == "get lettuce" \
                    or MACROACTIONNAME[macro_action] == "get plate 1" or MACROACTIONNAME[macro_action] == "get plate 2":
                        target_x, target_y = self._findPOitem(agent, macro_action)
                        macroAction2Item = {"get tomato": self.tomato[0], "get lettuce": self.lettuce[0], "get plate 1": self.plate[0], "get plate 2": self.plate[1]}
                        item = macroAction2Item[MACROACTIONNAME[macro_action]]
                        if target_x != item.x or target_y != item.y:
                                self.macroAgent[idx].cur_macro_action_done = True


    def _computeLowLevelActions(self, macro_actions):

        primitive_actions = []
        # loop each agent
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                self.macroAgent[idx].cur_macro_action = macro_actions[idx]
                macro_action = macro_actions[idx]
                self.macroAgent[idx].cur_macro_action_done = False
            else:
                macro_action = self.macroAgent[idx].cur_macro_action

            primitive_action = ACTIONIDX["stay"]

            if macro_action == 0:
                self.macroAgent[idx].cur_macro_action_done = True
            elif MACROACTIONNAME[macro_action] == "chop":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]] 
                    if new_name == "knife":
                        knife = self._findItem(new_x, new_y, new_name)
                        if isinstance(knife.holding, Food):
                            if not knife.holding.chopped:
                                primitive_action = action
                                self.macroAgent[idx].cur_chop_times += 1
                                if self.macroAgent[idx].cur_chop_times >= 3:
                                    self.macroAgent[idx].cur_macro_action_done = True
                                    self.macroAgent[idx].cur_chop_times = 0
                                break
                if primitive_action == ACTIONIDX["stay"]:
                    self.macroAgent[idx].cur_macro_action_done = True
            elif MACROACTIONNAME[macro_action] == "deliver" and agent.x == 1 and agent.y == 1 and ITEMNAME[agent.pomap[2][1]] == "agent":
                primitive_action = ACTIONIDX["right"]
            elif MACROACTIONNAME[macro_action] == "go to counter":
                findEmptyCounter = False
                target_x = 0
                target_y = int(self.ylen // 2)
                for i in self.counterSequence:
                    if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                        target_x = i
                        findEmptyCounter = True
                        break
                if findEmptyCounter:
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                else:
                    primitive_action = ACTIONIDX["stay"]
                    self.macroAgent[idx].cur_macro_action_done = True
            elif macro_action > 9:
                self.macroAgent[idx].cur_macro_action_done = True
                action = macro_action - 10
                new_x = agent.x + DIRECTION[action][0]
                new_y = agent.y + DIRECTION[action][1]
                if ITEMNAME[agent.pomap[new_x][new_y]] == "space":
                    primitive_action = action
                else:
                    primitive_action = ACTIONIDX["stay"]
            else:
                target_x, target_y = self._findPOitem(agent, macro_action)

                inPlate = False
                if MACROACTIONNAME[macro_action] == "get tomato" or MACROACTIONNAME[macro_action] == "get lettuce":
                    if target_x >= agent.x - 2 and target_x <= agent.x + 2 and target_y >= agent.y - 2 and target_y <= agent.y + 2:
                        for plate in self.plate:
                            if plate.x == target_x and plate.y == target_y:
                                primitive_action = ACTIONIDX["stay"]
                                self.macroAgent[idx].cur_macro_action_done = True
                                inPlate = True
                                break
                if inPlate:
                    primitive_actions.append(primitive_action)
                    continue
            
                if target_x == 1 and target_y == 0 and agent.x == 3 and agent.y == 1 and ITEMNAME[agent.pomap[2][1]] == "agent":
                    primitive_action = ACTIONIDX["right"]

                elif ITEMNAME[agent.pomap[target_x][target_y]] == "agent" \
                    and target_x >= agent.x - 2 and target_x <= agent.x + 2 and target_y >= agent.y - 2 and target_y <= agent.y + 2:
                    self.macroAgent[idx].cur_macro_action_done = True
                else:
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if primitive_action == ACTIONIDX["stay"]:
                        self.macroAgent[idx].cur_macro_action_done = True
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                        if (MACROACTIONNAME[macro_action] == "get plate 1"\
                        or MACROACTIONNAME[macro_action] == "get plate 2") and agent.holding:
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    self.macroAgent[idx].cur_macro_action_done = False
                                else:
                                    primitive_action = ACTIONIDX["stay"]
                        
                        if (MACROACTIONNAME[macro_action] == "go to knife 1"\
                        or MACROACTIONNAME[macro_action] == "go to knife 2") and not agent.holding:
                            primitive_action = ACTIONIDX["stay"]

                        if MACROACTIONNAME[macro_action] == "get tomato"\
                            or MACROACTIONNAME[macro_action] == "get lettuce":
                                for knife in self.knife:
                                    if knife.x == target_x and knife.y == target_y:
                                        if isinstance(knife.holding, Food):
                                            if not knife.holding.chopped:
                                                primitive_action = ACTIONIDX["stay"]
                                                break

                        if MACROACTIONNAME[macro_action] == "get tomato" or MACROACTIONNAME[macro_action] == "get lettuce" \
                            or MACROACTIONNAME[macro_action] == "get plate 1" or MACROACTIONNAME[macro_action] == "get plate 2":
                            macroAction2Item = {"get tomato": self.tomato[0], "get lettuce": self.lettuce[0], "get plate 1": self.plate[0], "get plate 2": self.plate[1]}
                            item = macroAction2Item[MACROACTIONNAME[macro_action]]
                            if target_x != item.x or target_y != item.y:
                                 primitive_action = ACTIONIDX["stay"]
 
                            

            primitive_actions.append(primitive_action)
        return primitive_actions
           
    # A star
    def _navigate(self, agent, target_x, target_y):

        direction = [(0,1), (0,-1), (1,0), (-1,0)]
        actionIdx = [0, 2, 1, 3]

        # make the agent explore up and down first to aviod deadlock when going to the knife
        
        q = PriorityQueue()
        q.put(AStarAgent(agent.x, agent.y, 0, self._calDistance(agent.x, agent.y, target_x, target_y), None, [], 0))
        isVisited = [[False for col in range(self.ylen)] for row in range(self.xlen)]
        isVisited[agent.x][agent.y] = True

        while not q.empty():
            aStarAgent = q.get()

            for action in range(4):
                new_x = aStarAgent.x + direction[action][0]
                new_y = aStarAgent.y + direction[action][1]
                new_name = ITEMNAME[agent.pomap[new_x][new_y]] 

                if not isVisited[new_x][new_y]:
                    init_action = None
                    if aStarAgent.action is not None:
                        init_action = aStarAgent.action
                    else:
                        init_action = actionIdx[action]

                    if new_name == "space" or new_name == "agent":
                        pass_agent = 0
                        if new_name == "agent":
                            pass_agent = 1
                        g = aStarAgent.g + 1
                        f = g + self._calDistance(new_x, new_y, target_x, target_y)
                        q.put(AStarAgent(new_x, new_y, g, f, init_action, aStarAgent.history_action + [actionIdx[action]], pass_agent))
                        isVisited[new_x][new_y] = True
                    if new_x == target_x and new_y == target_y:
                        if self.debug:
                            print("target_x, target_y", target_x, target_y)
                            print("agent.x, agent.y", agent.x, agent.y)
                            print("agent.history_action", aStarAgent.history_action + [actionIdx[action]])
                            print("final action", init_action)
                        return init_action
        #if no path is found, stay
        return ACTIONIDX["stay"]


                    
    def _calDistance(self, x, y, target_x, target_y):
        return abs(target_x - x) + abs(target_y - y)
    
    def _calItemDistance(self, agent, item):
        return abs(item.x - agent.x) + abs(item.y - agent.y)

    def _collectCurMacroActions(self):
        # loop each agent
        cur_mac = []
        for agent in self.macroAgent:
            cur_mac.append(agent.cur_macro_action)
        return cur_mac


    def _computeMacroActionDone(self):
        # loop each agent
        mac_done = []
        for agent in self.macroAgent:
            mac_done.append(agent.cur_macro_action_done)
        return mac_done

    def _get_macro_obs(self):
        macro_obs = []
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                obs = []
                for item in self.itemList:
                    x = 0
                    y = 0
                    if item.x >= agent.x - 2 and item.x <= agent.x + 2 and item.y >= agent.y - 2 and item.y <= agent.y + 2:
                        x = item.x / self.xlen
                        y = item.y / self.ylen
                        obs.append(x)
                        obs.append(y)
                        if isinstance(item, Food):
                            obs.append(item.cur_chopped_times / item.required_chopped_times)
                    else:
                        obs.append(0)
                        obs.append(0)
                        if isinstance(item, Food):
                            obs.append(0)
                obs += self.oneHotTask 
                self.macroAgent[idx].cur_macro_obs = obs 
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs))
        return macro_obs

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n