# Hamiltonian LLM

In this repository, we aim to train an alternative transformer-based model which based on Hamiltonian motion to predict the next token. Unlike traditional GPT family treating token as static with only position embedding, Hamiltonian LLM treats token as both positional and ``momentum''. This makes the model stick to the ``flow'' of the sentence and stay logical without being repititive or semantic drift. In addition, we introduce ``steering'' ability, which use the momentum vector to ``push'' the text steer into given topic specify by user.
In the model, we combine two components: A local convolution model used for capturing the grammar and local context of the text. This local model has a ``decay'' parameters, which force it to remember only recent text. The long-term connectivity and semantic context is processed by a transformer model that guide the potential. This division of role helps the model to converge faster because the Transformer model does not have to spend time to learn low level semantic rule.

The result: We end up with a base 124M parameters model which achieve loss \~3.4 after 10000 training iterations of effective batch size 128 and context window of size 1024 on Fine-web Edu. 

The weight of the base model can be found at:
[drive](https://drive.google.com/file/d/1v1y0pGEimN6rRQSu5Flez76LgulF3oA2/view?usp=sharing)

How to run test generation:
```
python test_mpc.py
```

How to run training for yourself:
```
python main.py
```
