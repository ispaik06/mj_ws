from gymnasium.envs.registration import register, registry

ENV_ID = "{{ENV_ID}}"

# 중복 등록 방지 (여러 번 import되어도 안전)
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="envs.{{ENV_MODULE}}:{{ENV_CLASS}}",
        max_episode_steps=1000,
    )
