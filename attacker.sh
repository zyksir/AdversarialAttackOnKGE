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
#    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/random_noise.py \
#        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/direct_addition.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
#    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/direct_addition.py \
#        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline --corruption_factor 20
#    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/instance_attribution.py \
#        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
#    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/fake_noise.py \
#        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
#    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/least_confidence.py \
#        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
#    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/least_confidence.py \
#        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline --corruption_factor 100
  done
done