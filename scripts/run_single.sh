\
source venv/Scripts/activate

tasks="0-15"
for dataset in acsincome 
do  
    for objective in extremile 
    do
        for optim in sgd dp_sgd
        do
            for epsilon in 2 1e8
            do
                python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8 --n_epochs 128 --dataset_length 4000 --epsilon $epsilon
            done
        done
    done
done

