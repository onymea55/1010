from gym.envs.registration import register

register(
    id='tenten-v0',
    entry_point='gym_tenten.envs:TentenEnv',
)