@echo off
cd C:\Users\guilh\Desktop\facial\GFPGAN

call .\venv\Scripts\activate

python inference_gfpgan.py -i C:\Users\guilh\Desktop\facial\clustering-de-images\recognized -o C:\Users\guilh\Desktop\facial\clustering-de-images\super_resolution -v 1.3 -s 4