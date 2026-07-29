@echo off
cd /d "%~dp0"
set HF_HOME=C:\Users\user\Desktop\product-studio-master\models\cache\hf_home
set HF_HUB_CACHE=C:\Users\user\Desktop\product-studio-master\models\cache
echo Started: %DATE% %TIME% > download_log.txt
echo Starting model download... (logs in download_log.txt)
venv\Scripts\python.exe -c "import os,sys; os.environ.update({'HF_HOME':r'C:\Users\user\Desktop\product-studio-master\models\cache\hf_home','HF_HUB_CACHE':r'C:\Users\user\Desktop\product-studio-master\models\cache'}); from huggingface_hub import snapshot_download; models=[('stabilityai/stable-diffusion-xl-base-1.0','SDXL 7GB'),('diffusers/controlnet-canny-sdxl-1.0','ControlNet 1.6GB'),('madebyollin/sdxl-vae-fp16-fix','VAE 0.3GB')]; [print(f'== {d} ==') or sys.stdout.flush() or snapshot_download(repo_id=r,local_dir=os.path.join(os.environ['HF_HUB_CACHE'],r),resume_download=True) or print(f'  {d} DONE') or sys.stdout.flush() for r,d in models]; print('ALL MODELS DOWNLOADED')" >> download_log.txt 2>&1
echo Done: %DATE% %TIME% >> download_log.txt
