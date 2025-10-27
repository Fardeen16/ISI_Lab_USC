# ISIS_Lab_USC


## PATH to project:
cd data/image_editing_project/ISI_NEW


## To make env:
conda create -y -n qwen310 python=3.10

## To activate env:
source ~/miniforge/bin/activate qwen310



## DELETE FILES:

single file:
rm filename.py

whole folder:
rm -rf old_folder_name


## QUERY:
python qwen.py --image images/1d2ksa9/image_1.jpg --out qwen_edited/out.png --prompt "remove the cannulae from her face." --steps 50


## CHECK STORAGE:
du -h  (All files storage)
du -h --max-depth=1 | sort -hr | head -20 (current folder space check)
