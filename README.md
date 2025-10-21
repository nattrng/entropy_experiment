# entropy_experiment
"Does having uncertainty in the output distribution indicate poorer performance on benchmarks?" Constrained testing in this scenario. All of the benchmarks (3) are done in a multiple-choice format.  This is not reflective of most use cases for these models. Ensuing approach would to use LLMs as a judge (perhaps via the Gemini API).

In this setting, entropy is referring to perplexity. Perplexity is not equivalent to shannon entropy.

1) conda create --name entropy_exp python=3.12.2
2) conda activate entropy_exp
3) git clone https://github.com/nattrng/entropy_experiment.git
4) pip install -r requirements.txt
5) python entropy_experiment.py

Just clone the repo and hit a quick "pip install -r requirements.txt".

Ensure that you have sufficient vram for your models.

make it so that it also lists # of times it got the weong answer rather than only displaying avg entropy
