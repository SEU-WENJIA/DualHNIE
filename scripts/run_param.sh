



for heads in 16; do
    for layers in 1; do
        for eta in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
        
        python run_dualhgnie_train.py \
        --dataset FB15k_rel_two \
        --data_path './datasets/fb15k_rel.pk' \
        --num-heads $heads \
        --num-layers $layers \
        --num-hidden 20 \
        --semantic_mode 'hyper_transformer' \
        --structure_mode 'hyper_attention'  \
        --eta $eta \
        --fusion 'fixed'

        done
    done
done    



for heads in 16; do
    for layers in 1; do
        for eta in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
        
        python run_dualhgnie_train.py \
        --dataset TMDB_rel_two \
        --data_path './datasets/tmdb_rel.pk'  \
        --num-heads $heads \
        --num-layers $layers \
        --num-hidden 20 \
        --semantic_mode 'hyper_transformer' \
        --structure_mode 'hyper_attention'  \
        --eta $eta   \
        --fusion 'fixed'

        done
    done
done    






for heads in 16; do
    for layers in 1; do
        for eta in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    
        python run_dualhgnie_train.py \
        --dataset MUSIC_rel_two \
        --data_path './datasets/music_rel.pk' \
        --num-heads $heads \
        --num-layers $layers \
        --num-hidden 20 \
        --semantic_mode 'hyper_transformer' \
        --structure_mode 'hyper_attention'  \
        --eta $eta \
        --fusion 'fixed'

        done
    done
done    

