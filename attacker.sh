for DATASET in FB15k-237 wn18rr
do
#  CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/eigen_attacker.py \
#       --init_checkpoint ./models/TransE_${DATASET}_baseline
#  cp ./models/TransE_${DATASET}_baseline/eigen.pkl ./models/RotatE_${DATASET}_baseline/
#  cp ./models/TransE_${DATASET}_baseline/centrality.pkl ./models/RotatE_${DATASET}_baseline/
#
#  cp ./models/TransE_${DATASET}_baseline/eigen.pkl ./models/DistMult_${DATASET}_baseline/
#  cp ./models/TransE_${DATASET}_baseline/centrality.pkl ./models/DistMult_${DATASET}_baseline/
#
#  cp ./models/TransE_${DATASET}_baseline/eigen.pkl ./models/ComplEx_${DATASET}_baseline/
#  cp ./models/TransE_${DATASET}_baseline/centrality.pkl ./models/ComplEx_${DATASET}_baseline/
  for MODEL in TransE RotatE DistMult ComplEx
  do
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/random_noise.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/direct_addition.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline --corruption_factor 10
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/instance_similarity.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/gradient_similarity.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/least_similarity.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/least_score.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline --corruption_factor 70  --num_cand_batch 2048
  done
done

DATASET=FB15k-237
for MODEL in TransE RotatE DistMult ComplEx
do
  CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/gradient_similarity.py \
      --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
done

DATASET=wn18rr
for MODEL in TransE RotatE DistMult ComplEx
do
  CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/gradient_similarity.py \
      --init_checkpoint ./models/${MODEL}_${DATASET}_baseline --num_cand_batch 8
done