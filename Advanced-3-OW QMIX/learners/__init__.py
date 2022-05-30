from .q_learner import QLearner
from .q_learner_w import QLearner as WeightedQLearner
from .max_q_learner import MAXQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["w_q_learner"] = WeightedQLearner
REGISTRY["max_q_learner"] = MAXQLearner

