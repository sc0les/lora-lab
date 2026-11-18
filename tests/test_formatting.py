import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FormattingTests(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType('datasets')
        stub.Dataset = object
        spec = importlib.util.spec_from_file_location('formatting_under_test', ROOT / 'utils/data_prep.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'datasets': stub}):
            spec.loader.exec_module(self.module)

    def test_empty_conversations(self):
        self.assertEqual(self.module.format_conversation_with_system_prompt('Brief.', []), [])

    def test_synthetic_turns_preserve_order(self):
        result = self.module.format_conversation_with_system_prompt('Brief.', [{
            'messages': [
                {'role': 'user', 'content': 'Question one'},
                {'role': 'assistant', 'content': 'Answer one'},
                {'role': 'user', 'content': 'Question two'},
            ],
        }])
        self.assertEqual(len(result), 1)
        text = result[0]['text']
        self.assertIn('<<SYS>>\nBrief.\n<</SYS>>', text)
        self.assertLess(text.index('Question one'), text.index('Answer one'))
        self.assertIn('[INST] Question two', text)


if __name__ == '__main__':
    unittest.main()
