# Week 0: External Coursework

The [preface](./preface.md#prerequisites) recommends two CMU courses. This page lists what each course offers as
hands-on work and how it relates to Week 0.

## CMU 10-714: Deep Learning Systems (Needle)

[Deep Learning Systems](https://dlsyscourse.org) is the course tiny-llm is modeled on. Its public homework repositories
come with starter code and local `pytest` suites. Two of them are directly useful before Week 1, and this repository
includes them as Git submodules so that they live inside your checkout:

| Path | Assignment | Pairs with |
|---|---|---|
| `prereqs/dlsys-hw0` | [Homework 0](https://github.com/dlsyscourse/hw0): NumPy warm-up, parsing MNIST, softmax regression, a two-layer network, and the same in C++ | Week 0 Days 1–3 |
| `prereqs/dlsys-hw1` | [Homework 1](https://github.com/dlsyscourse/hw1): the Needle automatic differentiation framework | Week 0 Day 4 |

The homework is linked as submodules rather than copied because those repositories do not publish a license. Fetch them
after cloning:

```bash
git submodule update --init prereqs/dlsys-hw0 prereqs/dlsys-hw1
# or clone everything at once
git clone --recurse-submodules <your-repo-url>
```

Each assignment is a notebook (`hw0.ipynb`, `hw1.ipynb`) written for Google Colab, but it also runs locally. The
assignments use a few extra packages; install them into a separate virtual environment so they stay out of the course
environment:

```bash
cd prereqs/dlsys-hw0
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest numdifftools pybind11
pip install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git

# Homework 0: fill in src/simple_ml.py, then run one question at a time
python3 -m pytest -k "add"
python3 -m pytest -k "parse_mnist"

# The C++ question: build the extension first (on macOS, see the note in the Makefile)
make
python3 -m pytest -k "softmax_regression_epoch_cpp"
```

```bash
cd prereqs/dlsys-hw1   # with the same virtual environment active
# Fill in python/needle/ops/ and python/needle/autograd.py, then
python3 -m pytest -l -v -k "forward"
python3 -m pytest -l -v -k "backward"
```

`mugrade` is only needed for the course's online submission; the tests import it but you do not need a grader key to run
them locally. Follow the notebook for the order of questions and for the explanations.

The rest of the course homework (hw2–hw4: neural network library, CPU/GPU array backends, and sequence models) is
optional for tiny-llm. Week 2 covers the GPU material in Metal instead of CUDA. The lecture videos and notes are on
[dlsyscourse.org](https://dlsyscourse.org/lectures/).

## CMU 10-601: Introduction to Machine Learning

[10-601](https://www.cs.cmu.edu/~mgormley/courses/10601/) publishes its lectures and readings, but its programming
assignments are handouts submitted to an autograder for enrolled students and are not published as a repository. Week 0
Days 1, 3, and 5 cover the parts of that course that tiny-llm relies on: linear algebra and numerics, neural networks
and backpropagation, and probabilistic models of sequences. Use the course's lecture notes as readings alongside those
days if you want the fuller treatment.

## Other Free Material Worth Doing

- [micrograd](https://github.com/karpathy/micrograd) and [minbpe](https://github.com/karpathy/minbpe), small codebases
  that pair with Days 4 and 5.
- [Neural Networks, Zero to Hero](https://karpathy.ai/zero-to-hero.html), whose "Let's build GPT" lecture is a good bridge
  from Day 6 to Week 1.

{{#include copyright.md}}
