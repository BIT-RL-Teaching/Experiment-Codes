from gym.envs.registration import register

register(
    id='Overcooked-v0',
    entry_point='macro_marl.env.overcooked:Overcooked',
)

register(
    id='Overcooked-PO-v0',
    entry_point='macro_marl.env.overcooked_PO_V0:POOvercooked_V0',
)

register(
    id='Overcooked-PO-MA-v0',
    entry_point='macro_marl.env.overcooked_PO_MA_V0:POOvercooked_MA_V0',
)

register(
    id='Overcooked-mapBC-PO-MA-v0',
    entry_point='macro_marl.env.overcooked_PO_MA_mapBC_V0:POOvercooked_MA_mapBC_V0',
)
