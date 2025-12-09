import json
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class DatasetLoader(ABC):
    """Abstract base class for loading and preprocessing QA datasets."""
    def __init__(self, file_path: str, num_samples: int = 10, seed: int = 42):
        self.file_path = file_path
        self.num_samples = num_samples
        random.seed(seed)
        self._samples = self._load_samples()
        self.contexts = self.get_contexts()
        self.queries = self.get_queries()
        self.answers = self.get_answers()

    def _load_raw_data(self) -> List[Dict]:
        """Load raw data from file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_samples(self) -> List[Dict]:
        raw_data = self._load_raw_data()
        n = min(self.num_samples, len(raw_data))
        return random.sample(raw_data, n)

    @abstractmethod
    def _sample_to_text(self, sample: Dict) -> str:
        """Convert a single sample to a plain text context string."""
        pass

    @abstractmethod
    def _get_answer(self, sample: Dict) -> str:
        """Extract the canonical answer string from a sample."""
        pass

    def get_contexts(self) -> List[str]:
        return [self._sample_to_text(sample) for sample in self._samples]

    def get_queries(self) -> List[str]:
        return [sample['question'] for sample in self._samples]

    def get_answers(self) -> List[str]:
        return [self._get_answer(sample) for sample in self._samples]

    def get_samples(self) -> List[Dict]:
        return self._samples


class HotpotQALoader(DatasetLoader):
    def _sample_to_text(self, sample: Dict) -> str:
        texts = []
        for title, sentences in sample['context']:
            para = f"## {title}\n" + "\n".join(sentences)
            texts.append(para)
        return "\n\n".join(texts)

    def _get_answer(self, sample: Dict) -> str:
        return sample['answer']


class GraphQuestionsLoader(DatasetLoader):
    def _sample_to_text(self, sample: Dict) -> str:
        nodes = {node['nid']: node for node in sample['graph_query']['nodes']}
        edges = sample['graph_query']['edges']
        desc_lines = []
        for edge in edges:
            start_name = nodes[edge['start']]['friendly_name']
            end_name = nodes[edge['end']]['friendly_name']
            rel_name = edge['friendly_name']
            desc_lines.append(f"{start_name} has {rel_name}: {end_name}")

        # Optionally add question node info
        q_nodes = [n for n in sample['graph_query']['nodes'] if n.get('question_node', 0) == 1]
        if q_nodes:
            q_name = q_nodes[0]['friendly_name']
            desc_lines.append(f"The question is about: {q_name}")

        return "\n".join(desc_lines)

    def _get_answer(self, sample: Dict) -> str:
        # GraphQuestions stores answer as a list; use first non-empty if exists
        answers = sample.get('answer', [])
        return answers[0] if answers else ""


# Factory function for convenience
def create_dataset_loader(
    file_path: str,
    dataset_type: Optional[str] = None,
    num_samples: int = 20,
    seed: int = 42
) -> DatasetLoader:
    if dataset_type is None:
        fname = file_path.lower()
        if 'hotpot' in fname:
            dataset_type = 'hotpotqa'
        elif 'graph' in fname and 'question' in fname:
            dataset_type = 'graphquestions'
        else:
            raise ValueError("Cannot auto-detect dataset type. Please specify `dataset_type`.")

    if dataset_type == 'hotpotqa':
        return HotpotQALoader(file_path, num_samples, seed)
    elif dataset_type == 'graphquestions':
        return GraphQuestionsLoader(file_path, num_samples, seed)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")