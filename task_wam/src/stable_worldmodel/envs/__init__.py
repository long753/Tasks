from gymnasium.envs.registration import register

register(
    id='swm/FourRoomDistractors-v0',
    entry_point='stable_worldmodel.envs.four_room.env:FourRoomDistractorsEnv',
)
