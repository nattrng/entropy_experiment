# entropy_experiment
"Does having uncertainty in the output distribution of language models indicate poorer performance on benchmarks?" 

Experimentation is constrained in this setting. I have not implemented other (possibly more enlightening) benchmarks; More crucially, ones unpertaining to multiple-choice formatted datasets. This was project was primarily a sanity check. What I have gleaned, however, is that entropy (shannon entropy) is not entirely indiciative of worse performance.

1) ```conda create --name entropy_exp python=3.12.2```
2) ```conda activate entropy_exp```
3) ```git clone https://github.com/nattrng/entropy_experiment.git```
4) ```pip install -r requirements.txt```
5) ```python entropy_experiment.py```

Ensure that you have sufficient VRAM for your models.

Sample Outputs
<img width="3030" height="1843" alt="image" src="https://github.com/user-attachments/assets/7a84749d-3f9c-410c-ae75-c79524d2a54c" />

<img width="3004" height="1843" alt="image" src="https://github.com/user-attachments/assets/9c31c31c-24f7-41f4-97aa-6a5e1959e121" />
