endpoints_dict = [
    dict(
        name="granite",
        description="Better for queries requiring reasoning, retrieving, and understanding.",
        type="hf",  # hf or api
        path="../../nvme/granite-4.0-micro",
        direct_results_path="work_dirs/granite4_4b_direct",
        meta_template="granite_4",
        max_new_tokens=512,
        cost=3.0,  # proxy: billions of parameters
    ),
    dict(
        name="qwen3",
        description="Better for general instruction following, reviewing, and reflecting.",
        type="hf",
        path="../../nvme/Qwen3-4B-Instruct-2507",
        direct_results_path="work_dirs/Qwen3-I_direct",
        meta_template="qwen3",
        max_new_tokens=512,
        cost=4.0,
    ),
]
