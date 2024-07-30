"""
Module for tag extraction using spaCy-LLM.
This module provides functionality to extract tags from text in multiple languages.
"""

import os
from typing import List, Optional, Dict
from pathlib import Path
import spacy
from spacy_llm.util import assemble
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the custom tag extractor
import tag_extractor

class ConfigLoader:
    """Handles loading of configuration files and associated resources."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "configs"
        self.example_dir = self.base_dir / "example"
        self.template_dir = self.base_dir / "templates"

    def get_paths(self, lang: str) -> Dict[str, str]:
        """Get the paths for configuration file and associated resources for a given language."""
        config_path = self.config_dir / f"config_{lang}.cfg"
        example_path = self.example_dir / f"{lang}_tags_few_shot.yaml"
        template_path = self.template_dir / f"tag_extractor_template_{lang}.jinja"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file for language '{lang}' not found.")
        
        return {
            "config": str(config_path),
            "example": str(example_path) if example_path.exists() else None,
            "template": str(template_path) if template_path.exists() else None
        }

class TagExtractor:
    """Main class for tag extraction."""

    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.nlp_models = {}

    def _load_model(self, lang: str) -> spacy.language.Language:
        """Load the spaCy model for a given language."""
        if lang not in self.nlp_models:
            paths = self.config_loader.get_paths(lang)
            config = spacy.util.load_config_from_str(open(paths["config"]).read())
            
            # Update paths in the config
            if paths["example"]:
                config["components"]["llm_tags"]["task"]["examples_path"] = paths["example"]
            if paths["template"]:
                config["components"]["llm_tags"]["task"]["template"] = paths["template"]
            
            self.nlp_models[lang] = spacy.util.load_model_from_config(config, auto_fill=True)
        return self.nlp_models[lang]

    def extract_tags(self, text: str, lang: str, n_tags: int) -> List[str]:
        """Extract tags from the given text."""
        nlp = self._load_model(lang)
        
        # Ensure the custom attribute is registered
        if not spacy.tokens.Doc.has_extension("article_tags"):
            spacy.tokens.Doc.set_extension("article_tags", default=None)
        
        # Update the n_tags in the pipeline's tag extractor component
        llm_component = nlp.get_pipe("llm_tags")
        llm_component.task.n_tags = n_tags
        
        doc = nlp(text)
        tags = doc._.get("article_tags")
        
        if tags is None:
            raise ValueError("Tag extraction failed. No tags were generated.")
        
        return tags

def initialize_environment():
    """Initialize environment variables and add project root to Python path."""
    load_dotenv()
    project_root = Path(__file__).parent.parent
    os.environ["PYTHONPATH"] = str(project_root)

def create_tag_extractor(base_dir: str = ".") -> TagExtractor:
    """Create and return a TagExtractor instance."""
    config_loader = ConfigLoader(base_dir)
    return TagExtractor(config_loader)

