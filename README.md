## TrackFormer + ByteTrack with VisDrone2019-MOT dataset

This is a repo that combines [TrackFormer](https://arxiv.org/abs/2101.02702) and [ByteTrack](https://arxiv.org/abs/2110.06864), the code is built on [TrackFormer](https://github.com/timmeinhardt/trackformer). 

The code can reach **50.3% MOTA and 66.2% IDF1** on VisDrone2019-MOT test  dataset, and can track object more stable compared to the model without ByteTrack:    
(The x axis is detection high thresh, = 0.4:0.1:0.9. 
In ByteTrack mode, I choose the low thresh = 0.5 and when the high thresh = 0.5 the low thresh = 0.4).

when the high thresh=0.6 the model reaches the best result.
 
 

![MOTA](https://github.com/JackWoo0831/trackformer-bytetrack/blob/master/imgs/MOTA.png)
![IDF1](https://github.com/JackWoo0831/trackformer-bytetrack/blob/master/imgs/IDF1.png)

***training details:*** I trained based on the pretrained model of Deformable DETR(r50_deformable_detr-checkpoint.pth) with 28 Epochs on 2 Tesla A100 GPUs and test on single A100 GPU. The initial lr is 1e-4 and 1e-5 of backbone. Optimizer is AdamW.

****model:**** Baidu Disk Link：https://pan.baidu.com/s/1fxzryW5NT3TEZL1XOX9XoQ 
code：zx0w




----
**TODO:**  
Now I'm trying multi-classes MOT on this code.

----
1. Installation:  
    The way to install is same as [TrackFormer Installation](https://github.com/timmeinhardt/trackformer/blob/main/docs/INSTALL.md)

2. Train on VisDrone-2019 MOT:  
    **Firstly** you should turn VisDrone to COCO format, 
    for train dataset, please run:  
    ```python
    python src/generate_coco_from_VisDrone.py  --split_name 'train_coco_all' --root_split 'VisDrone2019-MOT-train' 
    ```
   for val dataset:  
   ```python
   python src/generate_coco_from_VisDrone.py  --split_name 'train_coco_val' --root_split 'VisDrone2019-MOT-val' 
   ```
	for test dataset:  
   ```python
   python src/generate_coco_from_VisDrone.py  --split_name 'train_coco_test' --root_split 'VisDrone2019-MOT-test-dev' 
   ```	
     **Remember to change your dir in generate_coco_from_VisDrone.py accordingly, i.e. DATA_ROOT(line 25)**
		**Note:**
		So far The code can only work in a single class, that is "car", so I ignored the sequence which only contains pedestrian and people, and only filter the car class:  
		(see generate_coco_from_VisDrone.py line 167):   
   ```python
   if row[6] == '1' and row[7] in ['4']:  # score部分为1,评估时考虑边界框 并且目标的类别为车辆 
   ```	
		
	**Secondly** you can train the model based on Deformable DETR pretrained model (Backbone: ResNet50 in COCO detection):
   ```python
   python src/train.py with deformable tracking VisDrone full_res resume=models/r50_deformable_detr-checkpoint.pth output_dir=<your output path> epochs=40 lr_drop=10 
   ```	  
   
	**Or** you can continue train on my model:
   ```python
   python src/train.py with deformable tracking VisDrone full_res resume=models/checkpoint.pth output_dir=<your output path> epochs=40 lr_drop=10 
   ```	

3. Evaluate on VisDrone-2019 MOT:  
	run:  
   ```python
   python src/train.py with deformable tracking VisDrone full_res resume=<your model path> output_dir=<your output path> eval_only=True 
   ```		

4. Run demo:
  you can copy a VisDrone sequence to /data and rename it as "mydemo", then run:  
     ```python
   python src/track.py with dataset_name=DEMO data_root_dir=data/mydemo output_dir=data/mydemo2 write_images=pretty
   ```	  

5. On several GPUs:  
	For example, if you want to train on 2 GPUs, run:  
     ```python
   python -m torch.distributed.launch --nproc_per_node=2 --use_env src/train.py with deformable tracking VisDrone full_res resume=models/r50_deformable_detr-checkpoint.pth output_dir=<your model path> epochs=20 lr_drop=10
   ```	 	

6. If you don't want to use ByteTrack:  
   in src/track.py line 104:  
   modify
     ```python
   bytetrack = True
   ```	    

More details, please check "run_trackformer.txt".

