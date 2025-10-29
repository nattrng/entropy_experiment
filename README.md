# entropy_experiment
"Does having uncertainty in the output distribution indicate poorer performance on benchmarks?" Constrained testing in this scenario. All of the benchmarks (3) are done in a multiple-choice format.  This is not reflective of most use cases for these models. Ensuing approach would to use LLMs as a judge (perhaps via the Gemini API).

In this setting, entropy is referring to shannon entropy. It is not equivalent to perplexity which boasts

1) ```conda create --name entropy_exp python=3.12.2```
2) ```conda activate entropy_exp```
3) ```git clone https://github.com/nattrng/entropy_experiment.git```
4) ```pip install -r requirements.txt```
5) ```python entropy_experiment.py```

Ensure that you have sufficient VRAM for your models.
