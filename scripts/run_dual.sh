
for heads in 16; do
    for layers in 2; do
        for model in  dualhgcn dualhgat dualhgt; do #dualgt
    
            python run_dualhgnn_train.py \
            --dataset FB15k_rel_two \
            --data_path './datasets/fb15k_rel.pk' \
            --num-heads $heads \
            --num-layers $layers \
            --num-hidden 20 \
            --model $model 

            python run_dualhgnn_train.py \
            --dataset TMDB_rel_two \
            --data_path './datasets/tmdb_rel.pk'  \
            --num-heads $heads \
            --num-layers $layers \
            --num-hidden 20 \
            --model $model 




            python run_dualhgnn_train.py \
            --dataset MUSIC_rel_two \
            --data_path './datasets/music_rel.pk'  \
            --num-heads $heads \
            --num-layers $layers \
            --num-hidden 20 \
            --model $model 
            

            python run_dualhgnn_train.py \
            --dataset IMDB_S_rel_two \
            --data_path './datasets/imdb_s_rel.pk'  \
            --num-heads 8 \
            --num-layers 1 \
            --num-hidden 20 \
            --model $model 
            
        done
    done
done    



