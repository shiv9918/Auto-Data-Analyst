import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import pandas as pd
from pandas.testing import assert_frame_equal

from agents.cleaning_agent import run_cleaning_agent
from agents.llm_fallback import reset_shared_limit_message


KEEP_VALUES = {"outlier_strategy": "none", "encoding_method": "none"}
VERIFY = ("verify_dataset", {})
INSPECT = ("inspect_dataset", {})
FILL = ("fill_missing", {"strategy": "auto"})
SUMMARY = "Cleaning completed and verified."


def scripted_cleaner(calls, error=None):
    def kickoff(*, tools, **kwargs):
        registry = {tool.name: tool.to_structured_tool() for tool in tools}
        for name, args in calls:
            registry[name].invoke(args)
        if error:
            raise error
        return SUMMARY, "groq"
    return kickoff


class CleaningAgentTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.directory = Path(workspace.name)
        self.addCleanup(reset_shared_limit_message)
        reset_shared_limit_message()

    def write_csv(self, rows, name="input.csv"):
        path = self.directory / name
        path.write_text(rows, encoding="utf-8")
        return path

    def clean(self, path, calls, config=KEEP_VALUES, error=None):
        with patch("agents.cleaning_agent.kickoff_with_llm_fallback", side_effect=scripted_cleaner(calls, error)):
            return run_cleaning_agent(str(path), config)

    def test_missing_values_use_tool_and_keep_original_file(self):
        path = self.write_csv("id,amount\n1,2\n2,\n3,4\n4,12\n")
        original = path.read_bytes()
        cleaned, report = self.clean(path, [INSPECT, FILL, VERIFY])
        self.assertEqual(cleaned["amount"].tolist(), [2, 4, 4, 12])
        self.assertIn("Fill missing affected 1 rows and 1 columns", report)
        self.assertIn(SUMMARY, report)
        self.assertEqual(path.read_bytes(), original)

    def test_deduplication_precedes_imputation(self):
        path = self.write_csv("amount,category\n1,x\n,x\n1,x\n1,y\n5,z\n")
        cleaned, report = self.clean(path, [INSPECT, ("remove_duplicates", {}), FILL, VERIFY])
        self.assertEqual(cleaned.index.tolist(), [0, 1, 3, 4])
        self.assertEqual(cleaned.duplicated().sum(), 1)
        self.assertIn(SUMMARY, report)

    def test_outliers_can_be_capped_or_removed(self):
        path = self.write_csv("id,amount\n1,2\n2,2\n3,2\n4,2\n5,2\n6,100\n")
        for strategy, rows in (("cap", 6), ("remove", 5)):
            with self.subTest(strategy=strategy):
                cleaned, report = self.clean(
                    path, [INSPECT, ("handle_outliers", {"strategy": strategy}), VERIFY],
                    {**KEEP_VALUES, "outlier_strategy": strategy},
                )
                self.assertEqual(len(cleaned), rows)
                self.assertEqual(cleaned["amount"].max(), 2)
                self.assertIn(SUMMARY, report)

    def test_outlier_and_encoding_preferences_cannot_be_overridden(self):
        path = self.write_csv("id,amount,category\n1,2,x\n2,2,x\n3,2,y\n4,2,y\n5,2,z\n6,100,z\n")
        original = pd.read_csv(path)
        for call in (("handle_outliers", {"strategy": "remove"}), ("encode_categorical", {"method": "onehot"})):
            with self.subTest(tool=call[0]), self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
                cleaned, report = self.clean(path, [INSPECT, call, VERIFY])
                assert_frame_equal(cleaned, original)
                self.assertIn("deterministic cleaning rules", report)

    def test_clean_dataset_only_needs_inspection_and_verification(self):
        path = self.write_csv("id,amount\n1,10\n2,14\n3,20\n")
        cleaned, report = self.clean(path, [INSPECT, VERIFY])
        assert_frame_equal(cleaned, pd.read_csv(path))
        self.assertIn(SUMMARY, report)

    def test_verification_rejects_skipped_work(self):
        path = self.write_csv("id,amount\n1,4\n2,\n3,8\n")
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            cleaned, report = self.clean(path, [INSPECT, VERIFY])
        self.assertEqual(cleaned.loc[1, "amount"], 6)
        self.assertIn("deterministic cleaning rules", report)

    def test_late_or_repeated_mutation_triggers_fallback(self):
        path = self.write_csv("id,amount\n1,4\n1,4\n2,\n3,8\n")
        dedupe = ("remove_duplicates", {})
        for calls in ([INSPECT, FILL, dedupe, VERIFY], [INSPECT, dedupe, FILL, FILL, VERIFY]):
            with self.subTest(calls=calls), self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
                cleaned, report = self.clean(path, calls)
                self.assertEqual(len(cleaned), 3)
                self.assertIn("deterministic cleaning rules", report)

    def test_model_failure_discards_partial_tool_changes(self):
        path = self.write_csv("id,amount\n1,2\n2,\n3,4\n4,12\n")
        calls = [INSPECT, ("fill_missing", {"strategy": "mean"})]
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING") as logs:
            cleaned, report = self.clean(path, calls, error=RuntimeError("private diagnostic"))
        self.assertEqual(cleaned.loc[1, "amount"], 4)
        self.assertIn("deterministic cleaning rules", report)
        self.assertNotIn("private diagnostic", "\n".join(logs.output))

    def test_rate_limit_preserves_retry_message_and_cleans_data(self):
        path = self.write_csv("id,amount\n1,4\n2,\n3,8\n")
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            cleaned, report = self.clean(path, [INSPECT], error=RuntimeError("429 rate limit, try again in 3s"))
        self.assertEqual(cleaned.loc[1, "amount"], 6)
        self.assertIn("Groq limit exceeded. Use after 0h 0m 3s.", report)

    def test_missing_verification_triggers_fallback(self):
        path = self.write_csv("id,amount\n1,4\n2,\n3,8\n")
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            cleaned, report = self.clean(path, [INSPECT, FILL])
        self.assertEqual(cleaned.loc[1, "amount"], 6)
        self.assertIn("deterministic cleaning rules", report)

    def test_multiple_problems_and_all_selected_tools(self):
        path = self.write_csv(
            "id,amount,region,empty\n1,10, North ,\n2,,?,\n3,10,North,\n4,11,South,\n"
            "5,12,South,\n6,10,North,\n7,13,East,\n8,999,East,\n1,10, North ,\n,,,\n"
        )
        calls = [
            INSPECT, ("normalize_placeholders", {}), ("remove_empty_rows_and_columns", {}),
            ("remove_duplicates", {}), FILL, ("trim_whitespace", {}),
            ("handle_outliers", {"strategy": "cap"}), ("encode_categorical", {"method": "onehot"}), VERIFY,
        ]
        cleaned, report = self.clean(path, calls, {})
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            expected, _ = self.clean(path, [], {}, RuntimeError("offline"))
        assert_frame_equal(cleaned, expected)
        self.assertEqual(len(cleaned), 8)
        self.assertTrue({"region_East", "region_North", "region_South"} <= set(cleaned.columns))
        self.assertLess(cleaned["amount"].max(), 999)
        self.assertIn(SUMMARY, report)

    def test_mean_encoding_uses_selected_target_and_preserves_existing_columns(self):
        path = self.write_csv("category,amount,category_mean_encoded\nx,2,100\nx,4,200\ny,8,300\n")
        cleaned, report = self.clean(path, [INSPECT, ("encode_categorical", {"method": "mean"}), VERIFY],
                                     {**KEEP_VALUES, "encoding_method": "mean", "target_column": "amount"})
        self.assertEqual(cleaned["category_mean_encoded"].tolist(), [100, 200, 300])
        self.assertEqual(cleaned["category_mean_encoded_2"].tolist(), [3, 3, 8])
        self.assertIn(SUMMARY, report)

    def test_encoding_respects_cardinality_limit(self):
        path = self.write_csv("id,category\n1,x\n2,y\n3,z\n")
        cleaned, report = self.clean(path, [INSPECT, VERIFY],
                                     {**KEEP_VALUES, "encoding_method": "onehot", "max_categories": 2})
        assert_frame_equal(cleaned, pd.read_csv(path))
        self.assertIn(SUMMARY, report)

    def test_excel_uses_the_same_tools(self):
        path = self.directory / "input.xlsx"
        pd.DataFrame({7: [8, None, 12], 8: [1, 2, 3]}).to_excel(path, index=False)
        cleaned, report = self.clean(path, [INSPECT, FILL, VERIFY])
        self.assertEqual(cleaned[7].tolist(), [8, 10, 12])
        self.assertIn(SUMMARY, report)

    def test_concurrent_executions_keep_working_data_isolated(self):
        first = self.write_csv("id,amount\n1,1\n2,\n3,3\n", "one.csv")
        second = self.write_csv("id,amount\n1,20\n2,\n3,30\n", "two.csv")
        with patch("agents.cleaning_agent.kickoff_with_llm_fallback",
                   side_effect=scripted_cleaner([INSPECT, FILL, VERIFY])), ThreadPoolExecutor() as workers:
            outputs = list(workers.map(lambda path: run_cleaning_agent(str(path), KEEP_VALUES), [first, second]))
        self.assertEqual([df.loc[1, "amount"] for df, _ in outputs], [2, 25])
        self.assertTrue(all(SUMMARY in report for _, report in outputs))


class NativeToolCallingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(reset_shared_limit_message)
        reset_shared_limit_message()
        env = {
            "GROQ_API_KEY": "mock-only-credential", "CREW_MODEL": "groq/openai/gpt-oss-120b",
            "NO_PROXY": "*", "no_proxy": "*", "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
        }
        env_patch = patch.dict(os.environ, env)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.path = Path(workspace.name, "input.csv")
        self.path.write_text(
            "private_numeric_header,private_text_header\n4,private_value\n,private_value\n8,other\n",
            encoding="utf-8",
        )

    def run_http_script(self, calls, rate_limited=False):
        requests = []

        def respond(client, request, **kwargs):
            body = json.loads(request.read())
            requests.append((request.url.host, body))
            index = len(requests) - 1
            if rate_limited and index >= len(calls):
                return httpx.Response(429, request=request, headers={"Retry-After": "0"}, json={
                    "error": {"message": "Rate limit exceeded. Try again in 3s.",
                              "type": "tokens", "code": "rate_limit_exceeded"},
                })
            if index < len(calls):
                name, args = calls[index]
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{index}", "type": "function", "function": {
                        "name": name, "arguments": json.dumps(args),
                    }},
                ]}
                reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": SUMMARY}
                reason = "stop"
            return httpx.Response(200, request=request, json={
                "id": f"mock_{index}", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            })

        with patch.object(httpx.Client, "send", new=respond):
            cleaned, report = run_cleaning_agent(str(self.path), KEEP_VALUES)
        return cleaned, report, requests

    def test_native_calls_use_groq_send_json_and_keep_data_out_of_logs(self):
        with self.assertLogs("auto_data_analyst.cleaning", level="INFO") as logs:
            cleaned, report, requests = self.run_http_script([INSPECT, FILL, VERIFY])
        self.assertEqual(len(requests), 4)
        self.assertEqual(cleaned.iloc[1, 0], 6)
        self.assertIn(SUMMARY, report)
        for host, body in requests:
            self.assertEqual(host, "api.groq.com")
            self.assertEqual(body["model"], "openai/gpt-oss-120b")
            self.assertFalse(body["parallel_tool_calls"])
            self.assertEqual({tool["function"]["name"] for tool in body["tools"]}, {
                "inspect_dataset", "normalize_placeholders", "remove_empty_rows_and_columns",
                "remove_duplicates", "fill_missing", "trim_whitespace", "handle_outliers",
                "encode_categorical", "verify_dataset",
            })
            payload = json.dumps(body)
            self.assertNotIn("private_numeric_header", payload)
            self.assertNotIn("private_value", payload)
        observations = [
            json.loads(message["content"]) for message in requests[-1][1]["messages"] if message["role"] == "tool"
        ]
        self.assertEqual(observations[0]["pending"], {"fill_missing": 1})
        self.assertEqual(observations[1]["affected_rows"], 1)
        self.assertTrue(observations[2]["verified"])
        log = "\n".join(logs.output)
        self.assertIn("input={'strategy': 'auto'}", log)
        self.assertIn("'affected_columns': 1", log)
        self.assertNotIn("private", log)
        self.assertNotIn("mock-only-credential", log)

    def test_schema_invalid_native_arguments_fall_back(self):
        for args in ({"strategy": "unsupported"}, {"strategy": "auto", "extra": "unexpected"}, {}):
            with self.subTest(args=args), self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
                cleaned, report, requests = self.run_http_script([INSPECT, ("fill_missing", args)])
                self.assertEqual(len(requests), 2)
                self.assertEqual(cleaned.iloc[1, 0], 6)
                self.assertIn("deterministic cleaning rules", report)

    def test_unknown_native_tool_falls_back(self):
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            cleaned, report, requests = self.run_http_script([INSPECT, ("unregistered_tool", {})])
        self.assertEqual(len(requests), 2)
        self.assertEqual(cleaned.iloc[1, 0], 6)
        self.assertIn("deterministic cleaning rules", report)

    def test_repeated_native_tool_falls_back(self):
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            cleaned, report, requests = self.run_http_script([INSPECT, INSPECT])
        self.assertEqual(len(requests), 2)
        self.assertEqual(cleaned.iloc[1, 0], 6)
        self.assertIn("deterministic cleaning rules", report)

    def test_native_rate_limit_keeps_retry_message(self):
        with self.assertLogs("auto_data_analyst.cleaning", level="WARNING"):
            cleaned, report, requests = self.run_http_script([INSPECT], rate_limited=True)
        self.assertGreaterEqual(len(requests), 2)
        self.assertLessEqual(len(requests), 4)
        self.assertEqual(cleaned.iloc[1, 0], 6)
        self.assertIn("Groq limit exceeded. Use after 0h 0m 3s.", report)
