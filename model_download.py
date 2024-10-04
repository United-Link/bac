from huggingface_hub import hf_hub_download


model_path1 = hf_hub_download(
    repo_id="united-link/Bac_yolov9",  
    filename="best.pt",                      
    local_dir="./"            
)

print(f"Model downloaded to {model_path1}")

model_path2 = hf_hub_download(
    repo_id="united-link/Bac_yolov9",  
    filename="best_striped.pt",                      
    local_dir="./"            
)

print(f"Model downloaded to {model_path2}")
