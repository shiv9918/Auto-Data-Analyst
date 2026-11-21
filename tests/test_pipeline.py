import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest.mock import patch

import pandas as pd
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from agents.llm_fallback import get_shared_limit_message, reset_shared_limit_message
from main import run_pipeline
from tests.test_cleaning_agent import FILL, INSPECT, KEEP_VALUES, SUMMARY, VERIFY, scripted_cleaner


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        previous_directory = os.getcwd()
        self.addCleanup(os.chdir, previous_directory)
        os.chdir(workspace.name)

        patches = ExitStack()
        self.addCleanup(patches.close)
        self.eda, self.visualization, self.insight = [
            patches.enter_context(patch(f"agents.{name}_agent.kickoff_with_llm_fallback",
                                        return_value=("Verified fixture summary.", "groq")))
            for name in ("eda", "visualization", "insight")
        ]
        patches.enter_context(patch("agents.cleaning_agent.kickoff_with_llm_fallback",
                                    side_effect=scripted_cleaner([INSPECT, FILL, VERIFY])))

    def run_dataset(self, path):
        steps = []
        results = run_pipeline(str(path), on_step=steps.append, cleaning_config=KEEP_VALUES)
        self.assertEqual(list(results), [
            "clean_report", "eda_results", "viz_results", "insights", "report_path", "cleaned_data_path",
        ])
        self.assertEqual([message.split(" - ")[0] for message in steps[:5]],
                         [f"Step {number}/5" for number in range(1, 6)])
        self.assertEqual(len(steps), 6)
        self.assertTrue(steps[-1].startswith("Done! Report saved to:"))
        self.assertIn(SUMMARY, results["clean_report"])
        self.assertTrue(Path(results["report_path"]).read_bytes().startswith(b"%PDF"))
        csv_path = Path(results["cleaned_data_path"])
        self.assertEqual(csv_path.name, f"{path.stem}_cleaned.csv")
        self.assertTrue(results["viz_results"]["chart_paths"])
        for chart_path in results["viz_results"]["chart_paths"]:
            self.assertTrue(Path(chart_path).is_relative_to(csv_path.parent))
            self.assertTrue(Path(chart_path).is_file())
        self.assertTrue(Path(results["report_path"]).is_relative_to(csv_path.parent))
        return results

    def test_five_stages_produce_pdf_and_cleaned_csv_for_both_upload_formats(self):
        frame = pd.DataFrame({"id": [1, 2, 3], "amount": [4, None, 8]})
        frame.to_csv("input.csv", index=False)
        frame.to_excel("input.xlsx", index=False)
        for path in [Path("input.csv"), Path("input.xlsx")]:
            with self.subTest(source=path.suffix):
                result = self.run_dataset(path)
                self.assertEqual(pd.read_csv(result["cleaned_data_path"])["amount"].tolist(), [4, 6, 8])
                self.assertEqual(result["eda_results"]["stats"]["describe"]["amount"]["mean"], 6)
        for mock in (self.eda, self.visualization, self.insight):
            self.assertEqual(mock.call_count, 2)
            self.assertTrue(all("tools" not in call.kwargs for call in mock.call_args_list))

    def test_concurrent_runs_with_same_filename_keep_separate_downloads(self):
        sources = [Path("first/input.csv"), Path("second/input.csv")]
        for path, low, high in zip(sources, [4, 20], [8, 40]):
            path.parent.mkdir()
            path.write_text(f"id,amount\n1,{low}\n2,\n3,{high}\n", encoding="utf-8")

        with ThreadPoolExecutor() as pool:
            first, second = pool.map(self.run_dataset, sources)
        self.assertNotEqual(first["report_path"], second["report_path"])
        self.assertNotEqual(first["cleaned_data_path"], second["cleaned_data_path"])
        self.assertEqual(pd.read_csv(first["cleaned_data_path"])["amount"].tolist(), [4, 6, 8])
        self.assertEqual(pd.read_csv(second["cleaned_data_path"])["amount"].tolist(), [20, 30, 40])

    def test_failed_analysis_releases_only_its_own_artifacts(self):
        path = Path("input.csv")
        path.write_text("id,amount\n1,4\n2,\n3,8\n", encoding="utf-8")
        valid = self.run_dataset(path)
        with patch("agents.eda_agent.run_eda_agent", side_effect=ValueError("fixture error")):
            with self.assertRaisesRegex(ValueError, "fixture error"):
                run_pipeline(str(path), cleaning_config=KEEP_VALUES)
        valid_directory = Path(valid["cleaned_data_path"]).resolve().parent.relative_to(Path.cwd())
        self.assertEqual(list(Path("outputs").iterdir()), [valid_directory])
        self.assertTrue(Path(valid["report_path"]).is_file())

    def test_streamlit_analysis_produces_downloads_and_resets_on_replacement(self):
        def upload(file_id, data):
            return UploadedFile(UploadedFileRec(file_id, "input.csv", "text/csv", data),
                                FileURLs(file_id=file_id))

        app = AppTest.from_file(str(APP_PATH))
        with patch("streamlit.file_uploader", return_value=upload("first", b"id,amount\n1,4\n2,\n3,8\n")):
            app.run(timeout=15)
            app.radio[0].set_value("Leave as-is")
            app.radio[1].set_value("None")
            app.button[0].click().run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual({button.proto.label for button in app.get("download_button")},
                         {"⬇️ Download PDF Report", "⬇️ Download Cleaned Data (CSV)"})
        first_path = app.session_state["upload_path"]
        first_results = app.session_state["analysis_results"]
        self.assertIn(SUMMARY, first_results["clean_report"])
        self.assertEqual(app.session_state["dashboard_df"]["amount"].tolist(), [4, 6, 8])

        with patch("streamlit.file_uploader", return_value=upload("second", b"id,amount\n1,20\n2,\n3,40\n")):
            app.run(timeout=15)
        self.assertFalse(app.exception)
        self.assertNotEqual(first_path, app.session_state["upload_path"])
        self.assertIsNone(app.session_state["analysis_results"])


class SessionTests(unittest.TestCase):
    def test_retry_banners_are_isolated_across_concurrent_runs(self):
        barrier = Barrier(2)

        def get_message(seconds):
            reset_shared_limit_message()
            error = RuntimeError(f"429 rate limit, try again in {seconds}s")
            get_shared_limit_message(error)
            barrier.wait(timeout=5)
            return get_shared_limit_message(RuntimeError("429 rate limit"))

        with ThreadPoolExecutor(2) as workers:
            first, second = workers.map(get_message, [3, 9])
        self.assertIn("0h 0m 3s", first)
        self.assertIn("0h 0m 9s", second)

    def test_streamlit_keeps_csv_and_excel_uploader_and_ask_box(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        self.assertFalse(app.exception)
        uploader, = app.get("file_uploader")
        self.assertEqual(uploader.proto.label, "Upload your data file")
        self.assertEqual(set(uploader.proto.type), {".csv", ".xlsx", ".xls"})
        self.assertEqual(len(app.get("chat_input")), 1)
