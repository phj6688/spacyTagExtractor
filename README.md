# Tag Extraction Project with spaCy-LLM

## Overview

This project implements a robust tag extraction system using spaCy and Language Model (LLM) integration. It's designed to extract relevant tags from text in multiple languages, leveraging the power of large language models for accurate and context-aware tag generation.

## Features

- Multi-language support
- Configurable number of tags to extract
- Customizable templates for LLM prompts
- Few-shot learning with example data
- Caching mechanism for improved performance

## Requirements

- Python 3.7+
- spaCy
- spaCy-LLM
- PyYAML
- python-dotenv
- Jinja2

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tag-extraction-project.git
   cd tag-extraction-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Project Structure

```
.
├── spacy_main.py
├── tag_extractor.py
├── configs/
│   └── config_en.cfg
├── templates/
│   └── tag_extractor_template_en.jinja
├── example/
│   └── en_tags_few_shot.yaml
├── requirements.txt
├── .env
└── README.md
```

## Usage

To use the tag extraction system, create a script that initializes the environment, creates a `TagExtractor` instance, and calls the `extract_tags` method:

```python
from spacy_main import initialize_environment, create_tag_extractor

def main():
    initialize_environment()
    extractor = create_tag_extractor()

    text = "Your input text here..."
    lang = "en"
    n_tags = 10

    try:
        tags = extractor.extract_tags(text, lang, n_tags)
        print(f"Extracted tags: {tags}")
    except Exception as e:
        print(f"Error extracting tags: {str(e)}")

if __name__ == "__main__":
    main()
```

## Customization

You can customize the tag extraction process by modifying the configuration files, templates, and examples for each language. This allows you to adapt the system to different domains or specific requirements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- [spaCy](https://spacy.io/) for providing the core NLP functionality
- [OpenAI](https://openai.com/) for the GPT-3.5 language model

## Contact

For any queries or suggestions, please open an issue in the GitHub repository.
