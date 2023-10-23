cd ..


python main.py --experiment singleshot --pruner npb --compression 1 --init_mode ERK --alpha 0.01 --max_p 9 --beta 1 --model resnet20 --model-class lottery --post-epoch 160 --optimizer momentum --dataset cifar10 --pre-epoch 0 --lr 0.1

python main.py --experiment singleshot --pruner npb --compression 1 --init_mode ERK --alpha 0.01 --chunk_size 16 --max_p 6 --beta 2 --model resnet18 --model-class tinyimagenet --post-epoch 100 --optimizer momentum --dataset tiny-imagenet --pre-epoch 0 --lr 0.01