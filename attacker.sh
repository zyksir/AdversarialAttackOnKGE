for DATASET in FB15k-237 wn18rr
do
  for MODEL in TransE RotatE DistMult ComplEx
  do
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/random_noise.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/direct_addition.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/direct_addition.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline --corruption_factor 20
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/instance_attribution.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
    CUDA_VISIBLE_DEVICES=$1 python codes/noise_generator/fake_noise.py \
        --init_checkpoint ./models/${MODEL}_${DATASET}_baseline
  done
done

nohup bash 1.sh 1 & > tmp1.output
nohup bash 2.sh 2 & > tmp2.output