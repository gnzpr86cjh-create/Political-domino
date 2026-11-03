import io
import tempfile
import unittest
from pathlib import Path

from docx import Document
from reportlab.pdfgen import canvas

from political_domino import config, extraction, retrieval, storage, textmatch

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def make_pdf(pages):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line); y -= 15
        c.showPage()
    c.save()
    return buf.getvalue()


class ExtractionTests(unittest.TestCase):
    def test_pdf_keeps_page_numbers(self):
        data = (SAMPLES / "source1_tarvos_export_licensing_notice.pdf").read_bytes()
        r = extraction.extract_file("a.pdf", data)
        self.assertEqual([p["page"] for p in r["pages"]], [1, 2])
        self.assertIn("45 working days", r["pages"][0]["text"])
        self.assertIn("domestic processing quota", r["pages"][1]["text"])

    def test_docx_and_txt(self):
        d = Document(); d.add_paragraph("Hello political risk world. " * 10)
        t = d.add_table(rows=1, cols=2); t.rows[0].cells[0].text = "Cost"; t.rows[0].cells[1].text = "22 percent"
        buf = io.BytesIO(); d.save(buf)
        r = extraction.extract_file("x.docx", buf.getvalue())
        self.assertIsNone(r["pages"][0]["page"])
        self.assertIn("Cost | 22 percent", r["pages"][0]["text"])
        r = extraction.extract_file("x.txt", ("Tarvos licence " * 20).encode("cp1252"))
        self.assertIn("Tarvos licence", r["pages"][0]["text"])

    def test_error_cases(self):
        cases = {
            "empty.txt": b"",
            "ws.txt": b"  \n  ",
            "n.csv": b"a,b",
            "bad.pdf": b"nope",
            "scan.pdf": (SAMPLES / "error_cases" / "scanned_image_only.pdf").read_bytes(),
            "bad.docx": b"PK-not-a-docx",
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(extraction.ExtractionError):
                extraction.extract_file(name, data)

    def test_oversize_file_rejected(self):
        with self.assertRaises(extraction.ExtractionError):
            extraction.extract_file("big.txt", b"a" * (config.MAX_FILE_MB * 1024 * 1024 + 1))

    def test_very_long_text_is_truncated_with_warning(self):
        text = ("Licensing delays raise costs. " * 40000).encode()
        self.assertGreater(len(text), config.MAX_SOURCE_CHARS)
        r = extraction.extract_file("long.txt", text)
        self.assertLessEqual(r["n_chars"], config.MAX_SOURCE_CHARS)
        self.assertTrue(any("very long" in w for w in r["warnings"]))

    def test_partial_scanned_pdf_warns(self):
        data = make_pdf(["This is a text page with plenty of content. " * 6, ".", "Another real page with enough text to count. " * 6])
        r = extraction.extract_file("p.pdf", data)
        self.assertEqual([p["page"] for p in r["pages"]], [1, 3])
        self.assertTrue(any("page(s) 2" in w for w in r["warnings"]))

    def test_injection_scan_flags_but_does_not_remove(self):
        text = (SAMPLES / "error_cases" / "injection_source.txt").read_text()
        self.assertTrue(extraction.scan_for_injection(text))
        r = extraction.extract_file("i.txt", text.encode())
        self.assertIn("Ignore all previous instructions", r["pages"][0]["text"])
        self.assertFalse(extraction.scan_for_injection("Tarvos adds a licence step."))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.sources = []
        for i, name in enumerate(["source1_tarvos_export_licensing_notice.pdf",
                                  "source2_magnet_supply_market_briefing.pdf",
                                  "source3_industrial_automation_weekly.txt"], 1):
            r = extraction.extract_file(name, (SAMPLES / name).read_bytes())
            r["id"] = f"S{i}"
            self.sources.append(r)

    def test_chunks_have_stable_ids_pages_and_size(self):
        a = retrieval.chunk_sources(self.sources)
        b = retrieval.chunk_sources(self.sources)
        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])
        self.assertTrue(all(len(c.text) <= config.CHUNK_CHARS + 5 for c in a))
        self.assertTrue(any(c.page == 2 and c.source_id == "S1" for c in a))
        self.assertTrue(all(c.page is None for c in a if c.source_id == "S3"))
        self.assertEqual(len({c.chunk_id for c in a}), len(a))

    def test_ranking_finds_relevant_chunk(self):
        chunks = retrieval.chunk_sources(self.sources)
        top = retrieval.rank_chunks(chunks, "penalty clauses for late delivery to customers")[0][1]
        self.assertEqual(top.source_id, "S3")
        top = retrieval.rank_chunks(chunks, "how long do licences take to process")[0][1]
        self.assertEqual(top.source_id, "S1")

    def test_selection_respects_budget_and_covers_sources(self):
        chunks = retrieval.chunk_sources(self.sources)
        sel = retrieval.select_evidence(chunks, "supply licence price", char_budget=2500, min_per_source=1)
        self.assertEqual({c.source_id for c in sel}, {"S1", "S2", "S3"})
        self.assertLessEqual(sum(len(c.text) for c in sel), 2500 + config.CHUNK_CHARS * 3)


class TextMatchTests(unittest.TestCase):
    TEXT = "Licences are valid for six months and are issued for a fixed volume.\nUnused volume lapses at expiry\nand cannot be carried over."

    def test_exact_ignores_case_whitespace_quotes(self):
        r = textmatch.find_quote("UNUSED  volume lapses at expiry and cannot be carried over", self.TEXT)
        self.assertEqual(r["status"], "exact")
        self.assertIn("Unused volume lapses at expiry", r["excerpt"])
        r = textmatch.find_quote("“Licences are valid for six months”", self.TEXT)
        self.assertEqual(r["status"], "exact")
        self.assertEqual(r["excerpt"], "Licences are valid for six months")

    def test_ellipsis_requires_all_segments(self):
        ok = textmatch.find_quote("Licences are valid for six months ... cannot be carried over", self.TEXT)
        self.assertEqual(ok["status"], "exact")
        bad = textmatch.find_quote("Licences are valid for six months ... never be carried over anywhere", self.TEXT)
        self.assertEqual(bad["status"], "not_found")

    def test_approximate_and_fabricated(self):
        near = textmatch.find_quote("Licences are valid for 6 months and are issued for a fixed volume", self.TEXT)
        self.assertEqual(near["status"], "approximate")
        self.assertIn("six months", near["excerpt"])  # excerpt is the real text
        fake = textmatch.find_quote("Licences cost twelve thousand euros per shipment and expire weekly", self.TEXT)
        self.assertEqual(fake["status"], "not_found")
        self.assertEqual(textmatch.find_quote("too short", self.TEXT)["status"], "not_found")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = config.DATA_DIR
        config.DATA_DIR = Path(self.tmp.name)

    def tearDown(self):
        config.DATA_DIR = self._old
        self.tmp.cleanup()

    def test_roundtrip_sources_and_listing(self):
        a = storage.new_assessment("Test", "Acme", "Will X happen?")
        r = extraction.extract_file("s.txt", ("Some source text about licences. " * 10).encode())
        s = storage.add_source(a, r)
        self.assertEqual(s["id"], "S1")
        storage.add_source(a, dict(r))              # same filename replaces
        self.assertEqual([x["id"] for x in a["sources"]], ["S2"])
        storage.save(a)
        b = storage.load(a["id"])
        self.assertEqual(b["sources"][0]["filename"], "s.txt")
        self.assertEqual(storage.list_assessments()[0]["name"], "Test")
        storage.delete(a["id"])
        self.assertIsNone(storage.load(a["id"]))

    def test_rejects_path_traversal_ids(self):
        self.assertIsNone(storage.load("../../etc/passwd"))
        with self.assertRaises(ValueError):
            storage.delete("../x")


if __name__ == "__main__":
    unittest.main()
