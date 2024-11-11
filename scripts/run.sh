tasks="0-15"
dataset="acsincome"

for objective in extremile superquantile esrm erm
do
    python scripts/lbfgs.py --dataset $dataset --objective $objective
    for optim in sgd
    do
        for epsilon in 2 4 8 16 1024
        do
            for batch_size in 32 64 128
            do
                python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8 --n_epochs 128 --batch_size $batch_size --dataset_length 4000 --epsilon $epsilon
            done
        done
    done
done    
