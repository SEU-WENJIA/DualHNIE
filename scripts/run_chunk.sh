for fusion in adaptive ; do
    for heads in 16; do
        for layers in 1; do
            for eta in 0.2; do
                for chunk_size in 10 100 1000 10000; do
            
                python run_dualhgnie_train.py \
                --dataset FB15k_rel_two \
                --data_path './datasets/fb15k_rel.pk' \
                --num-heads $heads \
                --num-layers $layers \
                --num-hidden 20 \
                --semantic_mode 'hyper_transformer' \
                --structure_mode 'hyper_attention'  \
                --eta $eta \
                --fusion $fusion \
                --chunked_size $chunk_size

                done
            done
        done
    done    
done



for fusion in fixed ; do
    for heads in 16; do
        for layers in 1; do
            for eta in 0.2; do
                for chunk_size in 100 1000 10000 100000; do
            
                python run_dualhgnie_train.py \
                --dataset TMDB_rel_two \
                --data_path './datasets/tmdb_rel.pk'  \
                --num-heads $heads \
                --num-layers $layers \
                --num-hidden 20 \
                --semantic_mode 'hyper_transformer' \
                --structure_mode 'hyper_attention'  \
                --epochs 4 \
                --eta $eta \
                --fusion $fusion \
                --chunked_size $chunk_size \
                --cross-num 1

                done
            done
        done
    done    
done
