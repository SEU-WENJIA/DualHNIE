
for alpha in $(seq 0.3 0.2 0.9); do
    
    for beta in $(seq 0.4 0.2 1); do

        python g_trans_train.py \
        --dataset FB15k_rel_two \
        --data_path './datasets/fb15k_rel.pk' \
        --num-heads 16\
        --num-layers 1 \
        --num-hidden 20 \
        --semantic_mode 'hyper_transformer' \
        --structure_mode 'hyper_attention'  \
        --beta $beta \
        --alpha $alpha

        
    done    
done    






for alpha in $(seq 0.1 0.2 0.9); do

for beta in $(seq 0.2 0.2 1); do
    python g_trans_train.py \
    --dataset TMDB_rel_two \
    --data_path './datasets/tmdb_rel.pk'  \
    --num-heads 16\
    --num-layers 1 \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  \
    --alpha $alpha  \
    --beta $beta
done   
done    


# for beta in $(seq 0.2 0.2 1); do


#     python g_trans_train.py \
#     --dataset TMDB_rel_two \
#     --data_path './datasets/tmdb_rel.pk'  \
#     --num-heads 16\
#     --num-layers 1 \
#     --num-hidden 20 \
#     --semantic_mode 'hyper_transformer' \
#     --structure_mode 'hyper_attention'  \
#     --beta $beta
    
# done    



for alpha in $(seq 0.1 0.2 0.9); do
for beta in $(seq 0.2 0.2 1); do
    python g_trans_train.py \
    --dataset MUSIC_rel_two \
    --data_path './datasets/music_rel.pk' \
    --num-heads 16\
    --num-layers 1 \
    --num-hidden 20 \
    --semantic_mode 'hyper_transformer' \
    --structure_mode 'hyper_attention'  \
    --alpha $alpha   \
    --beta $beta
done    
done    





#     python g_trans_train.py \
#     --dataset MUSIC_rel_two \
#     --data_path './datasets/music_rel.pk' \
#     --num-heads 16\
#     --num-layers 1 \
#     --num-hidden 20 \
#     --semantic_mode 'hyper_transformer' \
#     --structure_mode 'hyper_attention'  \
#     --beta $beta
    
# done    






# for heads in 4 8; do
#     for layers in 3; do
    
#     python g_trans_train.py \
#     --dataset TMDB_rel_two \
#     --data_path './datasets/tmdb_rel.pk'  \
#     --num-heads $heads \
#     --num-layers $layers \
#     --num-hidden 20 \
#     --semantic_mode 'hyper_transformer' \
#     --structure_mode 'hyper_attention'  
    
#     # python g_trans_train.py \
#     # --dataset TMDB_rel \
#     # --data_path './datasets/tmdb_rel.pk'  \
#     # --num-heads $heads \
#     # --num-layers $layers \
#     # --num-hidden 20 \
#     # --semantic_mode 'hyper_transformer' \
#     # --structure_mode 'hyper_attention'  


#     done
# done    




# for layers in 3; do

#     for heads in 8; do

    
#     python g_trans_train.py \
#     --dataset IMDB_S_rel_two \
#     --data_path './datasets/imdb_s_rel.pk'  \
#     --num-heads $heads \
#     --num-layers $layers \
#     --num-hidden 20 \
#     --semantic_mode 'hyper_transformer' \
#     --structure_mode 'hyper_attention'  
    
#     # python g_trans_train.py \
#     # --dataset IMDB_S_rel \
#     # --data_path './datasets/imdb_s_rel.pk'  \
#     # --num-heads $heads \
#     # --num-layers $layers \
#     # --num-hidden 20 \
#     # --semantic_mode 'hyper_transformer' \
#     # --structure_mode 'hyper_attention'  


#     done
# done    




# for heads in 4 8 16; do
#     for layers in 3; do
    
#     python g_trans_train.py \
#     --dataset MUSIC_rel_two \
#     --data_path './datasets/music_rel.pk' \
#     --num-heads $heads \
#     --num-layers $layers \
#     --num-hidden 20 \
#     --semantic_mode 'hyper_transformer' \
#     --structure_mode 'hyper_attention'  
    
#     # python g_trans_train.py \
#     # --dataset TMDB_rel \
#     # --data_path './datasets/tmdb_rel.pk'  \
#     # --num-heads $heads \
#     # --num-layers $layers \
#     # --num-hidden 20 \
#     # --semantic_mode 'hyper_transformer' \
#     # --structure_mode 'hyper_attention'  


#     done
# done    






# python g_trans_train.py \
# --dataset MUSIC_rel_two \
# --data_path './datasets/music_rel.pk' \
# --num-heads 16 \
# --num-layers 3 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention' 


# python g_trans_train.py \
# --dataset TMDB_rel \
# --data_path './datasets/tmdb_rel.pk' \
# --num-heads 16 \
# --num-layers 2 \
# --num-hidden 20 \
# --semantic_mode 'hyper_transformer' \
# --structure_mode 'hyper_attention'
