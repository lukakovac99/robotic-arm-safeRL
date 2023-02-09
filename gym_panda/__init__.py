from gym.envs.registration import register

register(
    id='panda-v0',
    entry_point='gym_panda.envs:PandaEnv',
)

register(
    id='xarm-v0',
    entry_point='gym_panda.envs:XarmEnv',
)

register(
    id='kuka-v0',
    entry_point='gym_panda.envs:KukaEnv',
)

