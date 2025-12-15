from huggingface_hub import create_repo, HfApi

# 1. create repository
<<<<<<< HEAD
hf_name = "StarVLA/BEHAVIOR-QwenDual-Pretrained-720"
create_repo(hf_name, repo_type="model", exist_ok=True)
=======
hf_name = "StarVLA/LLaVA-OneVision-COCO"
create_repo(hf_name, repo_type="dataset", exist_ok=True)
>>>>>>> starVLA/starVLA

# 2. initialize API
api = HfApi()

# 3. upload large folder
<<<<<<< HEAD
folder_path = "/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/1_need/1205_BEHAVIOR_QwenDual_Pretrain_720"
=======
folder_path = "/mnt/petrelfs/yejinhui/Projects/starVLA/playground/Datasets/sharegpt4v_coco"
>>>>>>> starVLA/starVLA
# 4. use upload_large_folder to upload
api.upload_large_folder(folder_path=folder_path, repo_id=hf_name, repo_type="dataset")
