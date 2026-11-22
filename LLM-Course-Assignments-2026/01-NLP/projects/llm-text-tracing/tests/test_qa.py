import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qa.engine import REFUSAL, answer, note_hits, retrieve
from qa.index import EvidenceIndex, build
from rag_eval import measure


class QuoteFirst:
    def choose(self, prompt, passages):
        return 0, passages[0]["answer"], len(prompt)


class EvidenceQATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name) / "index"
        docs = [
            ("a", "fact", "A", "测试 A 字符基线联合得分", "测试 A 联合得分为 0.6706。", "local", "A 组结果。"),
            ("b", "fact", "B", "测试 B 字符基线联合得分", "测试 B 联合得分为 0.5548。", "local", "B 组结果。"),
            ("case-63", "case", "", "训练案例 63", "案例 63 的家族为 Google。", "local", "原文证据。"),
        ]
        with patch("qa.index.documents", return_value=iter(docs)):
            build("unused", folder)
        cls.index = EvidenceIndex(folder)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_case_followup_and_unknown_id(self):
        row = answer(self.index, QuoteFirst(), "案例63是什么家族？")
        self.assertTrue(row["verified"])
        self.assertEqual(answer(self.index, QuoteFirst(), "那它是哪一家的？", [row])["citation"], "case-63")
        self.assertEqual(answer(self.index, QuoteFirst(), "案例999的家族？", [row])["answer"], REFUSAL)

    def test_followup_overrides_previous_split(self):
        row = answer(self.index, QuoteFirst(), "测试 A 的联合得分？")
        self.assertEqual(row["citation"], "a")
        self.assertEqual(answer(self.index, QuoteFirst(), "那换成 B 呢？", [row])["citation"], "b")
        self.assertFalse(retrieve(self.index, "测试 B 四段模型误报几篇？", [], "", True))

    def test_personal_notes_offsets_and_quote_verification(self):
        notes = "其他内容。\n  档案批次代号是青竹。"
        query = "材料的档案批次代号是什么？"
        hit = note_hits(query, notes)[0]
        _, lo, hi = hit["id"].split("-")
        self.assertEqual(notes[int(lo):int(hi)], hit["answer"])
        row = answer(self.index, QuoteFirst(), query, notes=notes)
        self.assertEqual(row["answer"], hit["answer"])
        generator = QuoteFirst()
        generator.choose = lambda prompt, passages: (0, "未经证实的内容", 20)
        self.assertEqual(answer(self.index, generator, query, notes=notes)["answer"], REFUSAL)

    def test_calculation_separates_missing_citation_and_wrong_answer(self):
        turns = [
            {"expected": "a", "citation": "a", "correct": True, "verified": True, "retrieved": ["a"]},
            {"expected": "b", "citation": "", "correct": False, "verified": False, "retrieved": []},
            {"expected": "", "citation": "a", "correct": False, "verified": True, "retrieved": ["a"]},
        ]
        metrics = measure(turns)
        self.assertEqual(metrics["citation_micro_f1"], 0.5)
        self.assertEqual(metrics["unsupported_per_answer"], 0.5)
        self.assertEqual(metrics["answer_accuracy"], 1 / 3)
