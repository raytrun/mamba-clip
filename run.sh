python -m torch.distributed.launch --nnodes=2 --nproc_per_node=8 \
    main.py \
    --model CLIP_Simba_B --root ${HOME}/data/yfcc15m \
    --imagenet-val-zip ${HOME}/data/imagenet/val.zip \
    --imagenet-val-txt ${HOME}/data/imagenet/val_map.txt \
    --output-dir output/simba_b  \
    --batch-size 256 --lr 5e-4 --wd 0.5 --print-freq 100 \
    --workers 8 \
    --epochs 25 --wandb

