from gymnasium.envs import registration


WORLDS = set()
DISCRETE_WORLDS = set()


def register(id, entry_point, discrete=False, **kwargs):
    registration.register(id=id, entry_point=entry_point, **kwargs)
    WORLDS.add(id)
    if discrete:
        DISCRETE_WORLDS.add(id)


##############
# CONTINUOUS #
##############


register(
    id='swm/OGBCube-v0',
    entry_point='stable_worldmodel.envs.ogbench.cube_env:CubeEnv',
)

register(
    id='swm/OGBScene-v0',
    entry_point='stable_worldmodel.envs.ogbench.scene_env:SceneEnv',
)

register(
    id='swm/OGBMaze-v0',
    entry_point='stable_worldmodel.envs.ogbench.maze_env:MazeEnv',
)
