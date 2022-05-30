from functools import partial
from .multiagentenv import MultiAgentEnv
#from .grf import Academy_3_vs_1_with_Keeper, Academy_Counterattack_Hard
from .overcooked import OverCookedEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "overcooked" : partial(env_fn, env=OverCookedEnv),
    #"academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    #"academy_counterattack_hard": partial(env_fn, env=Academy_Counterattack_Hard),
}


