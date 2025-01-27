@echo off
cd C:\Users\guilh\Desktop\facial\CodeFormer

call .\venv\Scripts\activate

python inference_codeformer.py -w 0.5 --input_path C:\Users\guilh\Desktop\facial\clustering-de-images\recognized --output_path C:\Users\guilh\Desktop\facial\clustering-de-images\super_resolution
