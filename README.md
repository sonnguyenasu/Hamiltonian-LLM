# Hamiltonian LLM

In this repository, we aim to train an alternative transformer-based model which based on Hamiltonian motion to predict the next token. Unlike traditional GPT family treating token as static with only position embedding, Hamiltonian LLM treats token as both positional and "momentum". This makes the model stick to the "flow" of the sentence and stay logical without being repititive or semantic drift. In addition, we introduce "steering" ability, which use the momentum vector to "push" the text steer into given topic specify by user.
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

## Example of generated text:

Pull toward keyword generation:
```
Select Mode (1-4/q): 2
Enter Prompt: Hamiltonian LLM is
Target Word: quantum
Strength: 0.5
🧲 Steering towards 'quantum' with force 0.5...

Prompt: Hamiltonian LLM is
----------------------------------------
Hamiltonian LLM is the first to apply a quantum-statics theory of the interrelationship interaction between two objects. The experimenter and experimentalist, Cai Chengduming, conducted this research using a pairwise distribution model from both sets (coupled and field) in which the two particles interacting as one would interact at the same time (as illustrated by the presence of a strong attractor). The data were analyzed in order:
|=||–|g(A)=0−1,2
----------------------------------------
```

Push away from keyword generation:
```
Select Mode (1-4/q): 2
Enter Prompt: Hamiltonian LLM is
Target Word: quantum
Strength: -0.5
🧲 Steering towards 'quantum' with force -0.5...

Prompt: Hamiltonian LLM is
----------------------------------------
Hamiltonian LLM is based on the history of the country.
The first European settlers arrived in 1609 from South Africa, where they were given a free charter by King James I of England and became independent citizens after independence from France. In 1801 they returned to South America with their families in the U.S. From that time along the Atlantic coast there was an Indian confeder movement known as the The United LIVES (CRS). By the mid-19th century it had become Anglicized at a young
----------------------------------------
```

Normal generation
```
Select Mode (1-4/q): 1
Enter Prompt: Hamiltonian LLM is

Prompt: Hamiltonian LLM is
----------------------------------------
Hamiltonian LLM is a small, multi-disciplinary research program designed to assist the public in understanding and solving complex problems using the latest technologies. The goal of this project is to find out how to produce a high quality data source for future applications as well as develop computer solutions that are compatible with existing sources such as mobile phones or tablets.
The design phase was initially defined by the developer, who worked on developing an application based on the proposed Xcode architecture. This involves the development of the system file format (XML
----------------------------------------

```
