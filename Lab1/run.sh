#!/bin/bash


# ---------- 128x64 + Sigmoid ----------

python main.py --data=XOR --optimizer=SGD --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=linear --optimizer=SGD --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=XOR --optimizer=Adam --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=linear --optimizer=Adam --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=XOR --optimizer=SGD --lr=1e-4 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=linear --optimizer=SGD --lr=1e-4 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=XOR --optimizer=SGD --lr=1e-1 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

python main.py --data=linear --optimizer=SGD --lr=1e-1 --hidden_1=128 --hidden_2=64 --activation=Sigmoid

# ---------- 128x64 + Tanh ----------

python main.py --data=XOR --optimizer=SGD --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=Tanh

python main.py --data=linear --optimizer=SGD --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=Tanh

# ---------- 128x64 + SoftSign ----------

python main.py --data=XOR --optimizer=SGD --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=SoftSign

python main.py --data=linear --optimizer=SGD --lr=1e-2 --hidden_1=128 --hidden_2=64 --activation=SoftSign

# ---------- 32x16 + Sigmoid ----------

python main.py --data=XOR --optimizer=SGD --lr=1e-2 --hidden_1=32 --hidden_2=16 --activation=Sigmoid

python main.py --data=linear --optimizer=SGD --lr=1e-2 --hidden_1=32 --hidden_2=16 --activation=Sigmoid
