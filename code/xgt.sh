for i in `seq 1 5`;
do
    python3 xgt.py -d event -p $i -obj reg:linear
done
