
for heads in 4 8 16; do
    for layers in 1  2; do
    
    python run_dualhgnie_train.py \
    --dataset FB15k_rel_two \
    --data_path './datasets/fb15k_rel.pk' \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  
    

    python run_dualhgnie_train.py \
    --dataset FB15k_rel \
    --data_path './datasets/fb15k_rel.pk' \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  


    python run_dualhgnie_train.py \
    --dataset FB15k_rel_semantic \
    --data_path './datasets/fb15k_rel.pk' \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  

    done
done    





for heads in 4 8 16; do
    for layers in 1 2; do
    
    python run_dualhgnie_train.py \
    --dataset TMDB_rel_two \
    --data_path './datasets/tmdb_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  
    
    python run_dualhgnie_train.py \
    --dataset TMDB_rel \
    --data_path './datasets/tmdb_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  

    python run_dualhgnie_train.py \
    --dataset TMDB_rel_semantic \
    --data_path './datasets/tmdb_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  

    done
done    




for layers in 1 2; do
    for heads in 4 8 16; do

    python run_dualhgnie_train.py \
    --dataset MUSIC_rel_two \
    --data_path './datasets/music_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  
    
    python run_dualhgnie_train.py \
    --dataset MUSIC_rel \
    --data_path './datasets/music_rel.pk' \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  

    python run_dualhgnie_train.py \
    --dataset MUSIC_rel_semantic \
    --data_path './datasets/music_rel.pk' \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention' \
 

    done
done    



for layers in 1 2; do

    for heads in 4  8; do

    
    python run_dualhgnie_train.py \
    --dataset IMDB_S_rel_two \
    --data_path './datasets/imdb_s_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  
    
    python run_dualhgnie_train.py \
    --dataset IMDB_S_rel \
    --data_path './datasets/imdb_s_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  

    python run_dualhgnie_train.py \
    --dataset IMDB_S_rel_semantic \
    --data_path './datasets/imdb_s_rel.pk'  \
    --num-heads $heads \
    --num-layers $layers \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  

    done
done    





# python run_dualhgnie_train.py \
# --dataset FB15k_rel_semantic \
# --data_path './datasets/fb15k_rel.pk' \
# --num-heads 16 \
# --num-layers 1 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \




# python run_dualhgnie_train.py \
# --dataset FB15k_rel_two \
# --data_path './datasets/fb15k_rel.pk' \
# --num-heads 16 \
# --num-layers 1 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \









# python run_dualhgnie_train.py \
# --dataset TMDB_rel \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 1 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \


# python run_dualhgnie_train.py \
# --dataset TMDB_rel_semantic \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 1 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \



# python run_dualhgnie_train.py \
# --dataset TMDB_rel_concat \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 1 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \




# python run_dualhgnie_train.py \
# --dataset TMDB_rel_two \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 1 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \
# --gpu 0







# python run_dualhgnie_train.py \
# --dataset TMDB_rel \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 2 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \


# python run_dualhgnie_train.py \
# --dataset TMDB_rel_semantic \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 2 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \



# python run_dualhgnie_train.py \
# --dataset TMDB_rel_concat \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 2 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \




# python run_dualhgnie_train.py \
# --dataset TMDB_rel_two \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 2 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' \
# --gpu 0   \




