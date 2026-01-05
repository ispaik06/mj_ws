from gymnasium.envs.registration import register, registry

ENV_ID = "myhumanoid-v0"

# 중복 등록 방지 (여러 번 import되어도 안전)
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="envs.myhumanoid:MyhumanoidEnv",
        max_episode_steps=1000,
    )
