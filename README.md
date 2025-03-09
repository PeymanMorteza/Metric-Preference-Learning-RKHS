# Metric and Preference Learning with Reproducing Kernel Hilbert Spaces (RKHS)

This repository provides the official implementation of the paper:

"Representer Theorems for Metric and Preference Learning: Geometric Insights and Algorithms"
📌 Authored by Peyman Morteza (2025)

The codebase includes implementations of kernelized ideal point methods and comparative baselines, along with scripts to reproduce experimental results.

---

## 📊 Experimental Results

### Comparison of Ranking Inference Methods
The table below presents accuracy comparisons between different ranking inference methods on the Chameleon and Flatlizard datasets. For detailed methodology and discussion, please refer to the paper.
| Algorithm [^3]                          | Chameleon Data [^1] | FlatLizard Data [^2] |
|------------------------------------|---------------|----------------|
| [BT](https://www.jstor.org/stable/2334029)                                 | 0.83±0.03     | 0.86±0.06      |
| [BT-LR](https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf)                              | 0.71±0.03     | 0.84±0.04      |
| [BT-GP](https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf)                              | 0.75±0.04     | 0.80±0.05      |
| [RC](https://papers.nips.cc/paper_files/paper/2012/hash/9adeb82fffb5444e81fa0ce8ad8afe7a-Abstract.html)                                 | 0.61±0.06     | 0.66±0.05      |
| [RRC](https://proceedings.mlr.press/v124/jain20a/jain20a.pdf)                                | 0.61±0.03     | 0.66±0.01      |
| [SVD](https://proceedings.mlr.press/v51/cucuringu16.html)                                | 0.72±0.08     | 0.69±0.05      |
| [SVDC](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_5)                               | 0.65±0.06     | 0.81±0.05      |
| [SVDK](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_5)                               | 0.76±0.06     | 0.68±0.05      |
| [Serial](https://www.jmlr.org/papers/volume17/16-035/16-035.pdf)                             | 0.79±0.04     | 0.70±0.05      |
| [C-Serial](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_5)                           | 0.80±0.03     | 0.88±0.01      |
| [CC](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_5)                                 | 0.66±0.10     | 0.78±0.08      |
| [KCC](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_5)                                | 0.71±0.06     | 0.78±0.03      |
| Vanilla Ideal Point Method ([XD20](https://proceedings.neurips.cc/paper/2020/file/0561bc7ecba98e39ca7994f93311ba23-Paper.pdf),[MD21](https://jmlr.csail.mit.edu/papers/volume22/18-105/18-105.pdf) ,[CMVN22](https://proceedings.neurips.cc/paper_files/paper/2022/file/1fd4367793bcd3ad38a0b820fcc1b815-Paper-Conference.pdf))  | 0.66±0.07     | 0.70±0.05      |
| **Kernelelized Ideal Point Method (ours)**          | **TBA**     | **TBA**      |

## 🔧 Getting Started: Setting Up and Running Experiments

### Setting Up the Environment `ReperGeom`

To set up a virtual environment named `ReperGeom` and install the necessary dependencies, follow these steps:

1. **Create and Activate the Environment**  
   Open a terminal and run the following commands:

   ```bash
   python3 -m venv ReperGeom  # Create the virtual environment
   source ReperGeom/bin/activate  # Activate the environment
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running Experiments

### Experiments for Flatlizard and Chameleon Datasets

#### Using RKHS Method with Default Hyperparameters

To run experiments on the *Flatlizard* and *Chameleon* datasets using the RKHS method with default hyperparameters, execute:

```bash
bash scripts/run.sh --num_epochs 1000 --data_set chameleons --reg_lambda 0.002 --method RKHS
bash scripts/run.sh --num_epochs 1000 --data_set Flatlizard --reg_lambda 0.0001 --method RKHS
```

#### Using RKHS Method Without Regularization

To run the same experiments without regularization, use:

```bash
bash scripts/run.sh --num_epochs 1000 --data_set chameleons --reg_lambda 0.0 --method RKHS
bash scripts/run.sh --num_epochs 1000 --data_set Flatlizard --reg_lambda 0.0 --method RKHS
```

#### Using Vanilla Method

To run experiments with the vanilla method:

```bash
bash scripts/run.sh --num_epochs 1000 --data_set Flatlizard --method vanilla
```

### Experiments for Synthetic Data

#### Using RKHS Method with Circular Kernel

To run experiments on synthetic data using the RKHS method with a circular kernel:

```bash
bash scripts/run.sh --num_epochs 1000 --data_set synthetic --method RKHS --kernel circ --reg_lambda 0.007 --num_runs 3
```

#### Using RKHS Method with Circular Kernel Without Regularization

For the same experiments without regularization:

```bash
bash scripts/run.sh --num_epochs 1000 --data_set synthetic --method RKHS --kernel circ --reg_lambda 0.0 --num_runs 3
```

#### Using Vanilla Method

To run experiments on synthetic data using the vanilla method:

```bash
bash scripts/run.sh --num_epochs 1000 --data_set synthetic --method vanilla --num_runs 3
```
## 📄 Citation

```bibtex
  @article{TODO,
          title={Representer Theorems for Metric and Preference Learning: Geometric Insights and Practical Algorithms}, 
          author={Morteza, Peyman},
          journal={TODO},
          year={2025}
          }
```
## 📌 Footnotes

[^1]: **Chameleon Dataset** – Male Cape Dwarf Chameleons Contest dataset obtained from [SFFMW06](https://www.sciencedirect.com/science/article/pii/S0003347206001035).
[^2]: **Flatlizard Dataset** – Flatlizard competition dataset sourced from [WWK09](https://pmc.ncbi.nlm.nih.gov/articles/PMC2660994/).
[^3]: **Reported Results** – The results for methods other than ideal point methods are courtesy of [CCS22](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_5).



