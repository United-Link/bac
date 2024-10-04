from huggingface_hub import HfApi


api = HfApi()

api.upload_file(
    path_or_fileobj="best.pt", 
    path_in_repo="model.pt", 
    repo_id="LeoKE178/Yolov9_Poker",  
    commit_message="Upload model_striped.pt"
)

api.upload_file(
    path_or_fileobj="best_striped.pt", 
    path_in_repo="model_striped.pt", 
    repo_id="LeoKE178/Yolov9_Poker",  
    commit_message="Upload model_striped.pt"
)
