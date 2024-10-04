from huggingface_hub import hf_hub_download



model_path2 = hf_hub_download(
    repo_id="LeoKE178/Yolov9_Poker",  
    filename="model_striped.pt",                      
    local_dir="images"            
)

print(f"Model downloaded to {model_path2}")
