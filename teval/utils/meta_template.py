meta_template_dict = dict(
    internlm=[
        dict(role="system", begin="<|System|>:", end="\n"),
        dict(role="user", begin="<|User|>:", end="\n"),
        dict(role="function", begin="<|System|>:", end="\n"),
        dict(role="assistant", begin="<|Bot|>:", end="<eoa>\n", generate=True),
    ],
    llama2=[
        dict(role="system", begin="[INST]", end="[\INST]"),
        dict(role="user", begin="[INST]", end="[\INST]"),
        dict(role="function", begin="[INST]", end="[\INST]"),
        dict(role="assistant", begin="", end="</s>", generate=True),
    ],
    qwen=[
        dict(
            role="user", api_role="user", begin="\n<|im_start|>user\n", end="<|im_end|>"
        ),
        dict(
            role="system",
            api_role="system",
            begin="\n<|im_start|>user\n",
            end="<|im_end|>",
        ),
        dict(
            role="function",
            api_role="user",
            begin="\n<|im_start|>user\n",
            end="<|im_end|>",
        ),
        dict(
            role="assistant",
            api_role="assistant",
            begin="\n<|im_start|>assistant\n",
            end="<|im_end|>",
            generate=True,
        ),
    ],
    vicuna=[
        dict(role="user", begin="user: ", end="\n"),
        dict(role="system", begin="user: ", end="\n"),
        dict(role="function", begin="user: ", end="\n"),
        dict(role="assistant", begin="assistant: ", end="\n", generate=True),
    ],
    # we use chat APITemplateParser for chatglm due to some specific designs from ChatGLM
    chatglm=[
        dict(role="system", api_role="user"),
        dict(role="user", api_role="user"),
        dict(role="function", api_role="user"),
        dict(role="assistant", api_role="assistant", generate=True),
    ],
    phi3=[
        dict(role="system", begin="<|system|>\n", end="<|end|>\n"),
        dict(role="user", begin="<|user|>\n", end="<|end|>\n"),
        dict(role="function", begin="<|user|>\n", end="<|end|>\n"),
        dict(
            role="assistant",
            begin="<|assistant|>\n",
            end="<|end|>\n",
            generate=True,
        ),
    ],
    qwen2_5=[
        dict(role="system", begin="<|im_start|>system\n", end="<|im_end|>\n"),
        dict(role="user", begin="<|im_start|>user\n", end="<|im_end|>\n"),
        dict(
            role="assistant",
            begin="<|im_start|>assistant\n",
            end="<|im_end|>\n",
            generate=True,
        ),
        dict(
            role="tool",
            begin="<|im_start|>user\n<tool_response>\n",
            end="</tool_response><|im_end|>\n",
        ),
    ],
    gorilla_openfunctions_v2=[
        dict(
            role="system",
            begin="You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction: <<function>>",
            end="\n",
        ),
        dict(role="user", begin="<<question>>", end="\n"),
        dict(role="assistant", begin="### Response:", end="", generate=True),
    ],
    qwq=[
        dict(role="system", begin="<|im_start|>system\n", end="<|im_end|>\n"),
        dict(role="user", begin="<|im_start|>user\n", end="<|im_end|>\n"),
        dict(
            role="assistant",
            begin="<|im_start|>assistant\n",
            end="<|im_end|>\n",
            generate=True,
        ),
        dict(
            role="tool",
            begin="<|im_start|>user\n<tool_response>\n",
            end="</tool_response><|im_end|>\n",
        ),
    ],
)
