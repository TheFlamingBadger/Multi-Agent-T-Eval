endpoints_dict = [
    dict(
        name="granite",
        description="Granite 4.0 micro instruct-tuned checkpoint for concise, grounded responses.",
        type="hf",  # hf or api
        path="../../nvme/granite-4.0-micro",
        meta_template="granite_4",
        max_new_tokens=512,
    ),
    dict(
        name="qwen3",
        description="Qwen3 4B Instruct (2507) for general-purpose reasoning and conversation.",
        type="hf",
        path="../../nvme/Qwen3-4B-Instruct-2507",
        meta_template="qwen3",
        max_new_tokens=512,
    ),
]
