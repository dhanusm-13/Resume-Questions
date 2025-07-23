from huggingface_hub import HfApi, Repository

repo_id = "dhanusm-13/my-models"  # Your Hugging Face repo ID
local_dir = "./t5_finetuned_questions_final"  # Your local model folder

api = HfApi()
api.create_repo(repo_id=repo_id, private=False, exist_ok=True)

repo = Repository(local_dir=local_dir, clone_from=repo_id)
repo.push_to_hub()
