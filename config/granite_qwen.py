endpoints_dict = [
    dict(
        name="granite",
        description="Better for queries requiring reasoning, retrieving, and understanding.",
        type="hf",  # hf or api
        path="../../nvme/granite-4.0-micro",
        meta_template="granite_4",
        max_new_tokens=512,
    ),
    dict(
        name="qwen3",
        description="Better for general instruction following, reviewing, and reflecting.",
        type="hf",
        path="../../nvme/Qwen3-4B-Instruct-2507",
        meta_template="qwen3",
        max_new_tokens=512,
    ),
]
