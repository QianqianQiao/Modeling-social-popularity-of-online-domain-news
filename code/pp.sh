for i in `seq 1 5`;
do
    CUDA_VISIBLE_DEVICES=3 python3 neuralpp.py -p $i -m predict -model ../results/neuralpp/models/$i.pth
done
