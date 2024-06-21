MODEL_ALIAS_DICT = {
    "text-embedding-ada-002": "ada2",
    "text-embedding-3-small": "te3sm",
    "text-embedding-3-large": "te3lg"
}

def get_chroma_folder_name(model_name: str, chunk_size, chunk_overlap) -> str:
    return f"chroma_{MODEL_ALIAS_DICT[model_name]}_{chunk_size}-{chunk_overlap}"

def get_completions_folder_name(model_name: str) -> str:
    return f"completions_{MODEL_ALIAS_DICT[model_name]}"