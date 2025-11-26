endpoints_dict = [
    dict(
        name="azure_gpt4o",
        description="Azure GPT-4o; strongest general reasoning.",
        type="azure",
        env_path=".env",
        direct_results_path="work_dirs/azure_gpt4o_direct",
        cost=50.0,
    ),
    dict(
        name="granite4_4b",
        description="Granite 4.0 4B; lighter local model.",
        type="hf",
        path="../../nvme/granite-4.0-micro",
        direct_results_path="work_dirs/granite4_4b_direct",
        meta_template="granite_4",
        max_new_tokens=512,
        cost=3.0,
    ),
]
