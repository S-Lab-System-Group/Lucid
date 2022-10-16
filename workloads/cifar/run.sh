max=10

for (( i=1; i <= $max; ++i ))
do
   python profile_cifar_ddp.py
done
