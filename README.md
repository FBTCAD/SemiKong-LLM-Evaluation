# SemiKong-LLM-Evaluation
Python framework for SemiKong evaluation

## Overview
This repository contains the evaluation framework and datasets used to assess SemiKong, the first open-source Large Language Model (LLM) specifically designed for the semiconductor industry. The framework includes tools for dataset analysis, model inference testing, and performance evaluation.

## Repository Structure
```
semikong-eval/
├── scripts/
│   ├── datatojson.py      # Dataset loading and conversion script
│   ├── indexsearch.py     # JSON search and analysis tool
│   └── SKEval.py          # SemiKong evaluation script
├── datasets/
│   ├── semikong_train_entries_fixed.json    # Training dataset
│   ├── edatcad.json       # EDA/TCAD test cases
│   └── acronyms.json      # Semiconductor acronyms test cases
└── results/
    └── SKedatcad_short.txt    # Evaluation results
```

## Setup and Configuration

1. Configure the API settings in `SKEval.py`:
```python
CONFIG = {
    "api_url": "http://your-api-endpoint:1234/v1/chat/completions",
    "model_name": "semikong-70b",
    "max_tokens": 150,
    "temperature": 0.3,
    "fuzzy_threshold": 60,
    "keyword_threshold": 0.6,
}
```

## Usage

### Dataset Analysis
```bash
# Convert and analyze dataset
python scripts/datatojson.py

# Search within dataset
python scripts/indexsearch.py semikong_train_entries_fixed.json "search_term" output.json
```

### Model Evaluation
```bash
# Run evaluation
python scripts/SKEval.py
```

## Evaluation Metrics
The framework evaluates responses based on six key metrics:
- Clarity and Directness (C&D)
- Practicality and Immediate Usability (PIU)
- Efficiency and Brevity (E&B)
- Logical Flow and Coherence (LFC)
- Expert-to-Expert Communication (EEC)
- Use of Examples and Specificity (UES)

## Contributing
We welcome contributions to improve the evaluation framework. Please submit pull requests or open issues for any bugs or enhancements.

## Citation
If you use this evaluation framework in your research, please cite:
```bibtex
@article{benistant2024semikong,
  title={SemiKong Evaluation Framework: Testing and Analysis of the First Semiconductor Industry-specific LLM},
  author={Benistant, Francis},
  year={2024},
  email={flb@mltma.com},
  url={https://www.linkedin.com/in/francis-benistant-14882727/}
}
```

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments
- AI Alliance (https://thealliance.ai)
- Aitomatic
- Tokyo Electron
- FPT Software

## Contact
For questions or support, please open an issue in the GitHub repository.
