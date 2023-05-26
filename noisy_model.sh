for NOISE in grad_is_cos grad_is_dot grad_is_l2 grad_gs_cos grad_gs_dot grad_gs_l2 direct central_diff direct_rel direct_10 central_diff_10 g_rand l_rand is_cos is_dot is_l2 gs_cos gs_dot gs_l2 least_score_global least_score_local least_similar_cos least_similar_dot least_similar_l2 
do
  bash run.sh train RotatE FB15k-237 $1 baseline 1024 256 200 9.0 1.0 0.0005 50000 16 -de \
    --warm_up_steps 20000  --fake ${NOISE} --no_save
  bash run.sh train RotatE wn18rr $1 baseline 512 1024 200 6.0 0.5 0.0005 40000 8 -de \
    --warm_up_steps 20000  --fake ${NOISE} --no_save

  bash run.sh train TransE FB15k-237 $1 baseline 1024 256 200 9.0 1.0 0.0005 50000 16 \
    --warm_up_steps 20000  --fake ${NOISE} --no_save
  bash run.sh train TransE wn18rr $1 baseline 512 1024 200 6.0 0.5 0.0005 40000 8 \
    --warm_up_steps 20000  --fake ${NOISE} --no_save

  bash run.sh train ComplEx FB15k-237 $1 baseline 1024 256 200 200.0 1.0 0.001 100000 16 \
    -de -dr -r 0.00001  --fake ${NOISE} --no_save
  bash run.sh train ComplEx wn18rr $1 baseline 512 1024 200 200.0 1.0 0.002 80000 8 \
    -de -dr -r 0.000005  --fake ${NOISE} --no_save

  bash run.sh train DistMult FB15k-237 $1 baseline 1024 256 200 200.0 1.0 0.001 100000 16 \
    -r 0.00001  --fake ${NOISE} --no_save
  bash run.sh train DistMult wn18rr $1 baseline 512 1024 200 200.0 1.0 0.002 80000 8 \
    -r 0.000005  --fake ${NOISE} --no_save
done

