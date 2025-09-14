for fusion in fixed gate adaptive attention concat ; do
    for heads in 16; do
        for layers in 1; do
            for eta in 0.2; do
            
            python run_dualhgnie_train.py \
            --dataset FB15k_rel_two \
            --data_path './datasets/fb15k_rel.pk' \
            --num-heads $heads \
            --num-layers $layers \
            --num-hidden 20 \
            --semantic_mode 'hyper_transformer' \
            --structure_mode 'hyper_attention'  \
            --eta $eta \
            --fusion $fusion

            done
        done
    done    

done




for fusion in fixed gate adaptive attention concat ; do
    for heads in 16; do
        for layers in 1; do
            for eta in 0.2; do
            
            python run_dualhgnie_train.py \
            --dataset TMDB_rel_two \
            --data_path './datasets/tmdb_rel.pk'  \
            --num-heads $heads \
            --num-layers $layers \
            --num-hidden 20 \
            --semantic_mode 'hyper_transformer' \
            --structure_mode 'hyper_attention'  \
            --eta $eta \
            --fusion $fusion

            done
        done
    done    

done



for fusion in fixed gate adaptive attention concat ; do
    for heads in 16; do
        for layers in 1; do
            for eta in 0.2; do
            
            python run_dualhgnie_train.py \
            --dataset MUSIC_rel_two \
            --data_path './datasets/music_rel.pk' \
            --num-heads $heads \
            --num-layers $layers \
            --num-hidden 20 \
            --semantic_mode 'hyper_transformer' \
            --structure_mode 'hyper_attention'  \
            --eta $eta \
            --fusion $fusion

            done
        done
    done    

done



