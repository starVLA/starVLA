from huggingface_hub import HfApi, create_repo

# 1. create repository
hf_name = "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
create_repo(hf_name, repo_type="model", exist_ok=True)

# 2. initialize API
api = HfApi()

# 3. upload large folder
folder_path = "./results/Checkpoints/0427_oxe_bridge_rt_1_QwenPI_v3"
api.upload_large_folder(folder_path=folder_path, repo_id=hf_name, repo_type="model")
