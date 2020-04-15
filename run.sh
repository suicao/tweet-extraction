python train.py  --model roberta --batch_size 32 --accumulation_steps 3 --fold 0 --seed 13 --epochs 3 --train_full
python train.py  --model xlmr --batch_size 32 --accumulation_steps 3 --fold 0 --seed 23 --epochs 3 --train_full
python train.py  --model roberta-large --batch_size 8 --accumulation_steps 12 --fold 0 --seed 33 --epochs 3 --train_full
python train.py  --model roberta-detector --batch_size 8 --accumulation_steps 12 --fold 0 --seed 43 --epochs 3 --train_full
python train.py  --model xlmr-large --batch_size 4 --accumulation_steps 12 --fold 0 --seed 53 --epochs 3 --train_full
