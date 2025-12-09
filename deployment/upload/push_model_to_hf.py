from huggingface_hub import create_repo, HfApi

# 1. create repository
hf_name = "StarVLA/LLaVA-OneVision-COCO"
create_repo(hf_name, repo_type="dataset", exist_ok=True)

# 2. initialize API
api = HfApi()

# 3. upload large folder
folder_path = "/mnt/petrelfs/yejinhui/Projects/starVLA/playground/Datasets/sharegpt4v_coco"
# 4. use upload_large_folder to upload
api.upload_large_folder(folder_path=folder_path, repo_id=hf_name, repo_type="dataset")
