"""Module for extracting tags from text using LLM."""

from typing import List, Iterable, Optional, Dict, Any
from abc import ABC, abstractmethod
import os
import json
import yaml
from jinja2 import Template
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.ty import TaskResponseParser

class TemplateLoader(ABC):
    """Abstract base class for template loading."""

    @abstractmethod
    def load(self, path: str) -> str:
        """Load template from path."""

class FileTemplateLoader(TemplateLoader):
    """Concrete implementation of TemplateLoader for file-based templates."""

    def load(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template file not found: {path}")
        with open(path, 'r') as file:
            return file.read()

class ExamplesLoader(ABC):
    """Abstract base class for examples loading."""

    @abstractmethod
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load examples from path."""

class YAMLExamplesLoader(ExamplesLoader):
    """Concrete implementation of ExamplesLoader for YAML-based examples."""

    def load(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Examples file not found: {path}")
        with open(path, 'r') as file:
            return yaml.safe_load(file)

class ResponseParser(ABC):
    """Abstract base class for parsing LLM responses."""

    @abstractmethod
    def parse(self, response: str, n_tags: int) -> List[str]:
        """Parse the LLM response."""

class JSONResponseParser(ResponseParser):
    """Concrete implementation of ResponseParser for JSON responses."""

    def parse(self, response: str, n_tags: int) -> List[str]:
        try:
            tags = json.loads(response[0])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("Response is not a list of strings")
            if len(tags) != n_tags:
                raise ValueError(f"Expected {n_tags} tags, but got {len(tags)}")
            return tags
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Failed to parse response as JSON: {response}") from e

class TagExtractorTask:
    """Main class for tag extraction task."""

    def __init__(self, n_tags: int, template_loader: TemplateLoader, 
                 response_parser: ResponseParser, template_path: Optional[str] = None, 
                 examples: Optional[List[Dict[str, Any]]] = None):
        self.n_tags = n_tags
        self.template_loader = template_loader
        self.response_parser = response_parser
        self.template_path = template_path
        self.template = self._load_template()
        self.examples = examples

    def _load_template(self) -> Template:
        if self.template_path:
            template_str = self.template_loader.load(self.template_path)
            return Template(template_str)
        return Template(self._default_template())

    def _default_template(self) -> str:
        return ("Extract exactly {n_tags} tags from the following text. "
                "Provide only the tags as a JSON array of strings. "
                "For example: ['tag1', 'tag2', 'tag3']:\n\n{text}")
    
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        """Generate prompts for the given documents."""
        for doc in docs:
            yield self.template.render(text=doc.text, n_tags=self.n_tags, examples=self.examples)

    def parse_responses(self, docs: Iterable[Doc], responses: Iterable[str]) -> Iterable[Doc]:
        """Parse responses and update documents with extracted tags."""
        for doc, response in zip(docs, responses):
            try:
                tags = self.response_parser.parse(response, self.n_tags)
                doc._.set("article_tags", tags)
            except ValueError as e:
                # Log the error and set an empty list of tags
                print(f"Error parsing response for doc {doc.text[:50]}...: {str(e)}")
                doc._.set("article_tags", [])
            yield doc

    @property
    def prompt_template(self) -> str:
        """Return the prompt template string."""
        return self.template.render(text="{text}", n_tags=self.n_tags, examples=self.examples)

@registry.llm_tasks("spacy-tag-extractor.TagExtractor.v1")
def make_tag_extractor(n_tags: int = 10, template: Optional[str] = None, examples_path: Optional[str] = None) -> TagExtractorTask:
    """Factory function to create TagExtractorTask instances."""
    template_loader = FileTemplateLoader()
    response_parser = JSONResponseParser()
    examples = YAMLExamplesLoader().load(examples_path) if examples_path else None
    return TagExtractorTask(n_tags, template_loader, response_parser, template_path=template, examples=examples)

@registry.misc("spacy-tag-extractor.TagExtractorResponseParser.v1")
def make_tag_extractor_response_parser() -> TaskResponseParser:
    """Factory function to create TaskResponseParser instances."""
    def parser(docs, responses):
        extractor = TagExtractorTask(10, FileTemplateLoader(), JSONResponseParser())
        return extractor.parse_responses(docs, responses)
    return parser