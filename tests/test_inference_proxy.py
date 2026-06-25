"""
tests/test_inference_proxy.py

Unit tests for inference proxy — mocks Ollama and Bedrock.
"""

import sys
import json
import unittest
from unittest.mock import patch, MagicMock, Mock
import urllib.error

sys.path.insert(0, "lambda/inference_proxy")


class TestInferenceProxy(unittest.TestCase):

    def setUp(self):
        import os
        os.environ["ENVIRONMENT"] = "test"
        os.environ["OLLAMA_ENDPOINT"] = "http://localhost:11434"
        os.environ["ECS_CLUSTER"] = "test-cluster"
        os.environ["ECS_SERVICE"] = "test-service"
        os.environ["SSM_PREFIX"] = "/aws-hybrid-llm/test"

    @patch("handler.ssm")
    @patch("handler.ecs")
    @patch("handler.urllib.request.urlopen")
    def test_successful_local_inference(self, mock_urlopen, mock_ecs, mock_ssm):
        """Local Ollama returns answer — no fallback triggered."""
        # Mock SSM config
        mock_ssm.get_parameters_by_path.return_value = {
            "Parameters": [
                {"Name": "/aws-hybrid-llm/test/default-model", "Value": "llama3.2"},
                {"Name": "/aws-hybrid-llm/test/bedrock-fallback-enabled", "Value": "true"},
                {"Name": "/aws-hybrid-llm/test/local-timeout-ms", "Value": "25000"},
                {"Name": "/aws-hybrid-llm/test/bedrock-fallback-model",
                 "Value": "anthropic.claude-3-haiku-20240307-v1:0"},
            ]
        }

        # Mock ECS service running
        mock_ecs.describe_services.return_value = {
            "services": [{"desiredCount": 1, "runningCount": 1}]
        }

        # Mock Ollama response
        ollama_response = json.dumps({
            "message": {"content": "S3 is an object storage service."},
            "model": "llama3.2",
            "prompt_eval_count": 10,
            "eval_count": 20
        }).encode()

        mock_cm = MagicMock()
        mock_cm.__enter__ = Mock(return_value=Mock(read=Mock(return_value=ollama_response)))
        mock_cm.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_cm

        import handler
        handler._config_cache = {}  # Reset cache

        result = handler.lambda_handler(
            {"action": "infer", "query": "What is S3?"},
            {}
        )

        self.assertEqual(result["source"], "local")
        self.assertFalse(result["fallback_triggered"])
        self.assertIn("S3", result["answer"])

    @patch("handler.ssm")
    @patch("handler.ecs")
    @patch("handler.cloudwatch")
    @patch("handler.bedrock_runtime")
    @patch("handler.urllib.request.urlopen")
    def test_fallback_on_timeout(self, mock_urlopen, mock_bedrock, mock_cw,
                                  mock_ecs, mock_ssm):
        """Local Ollama times out — Bedrock fallback triggered."""
        mock_ssm.get_parameters_by_path.return_value = {
            "Parameters": [
                {"Name": "/aws-hybrid-llm/test/default-model", "Value": "llama3.2"},
                {"Name": "/aws-hybrid-llm/test/bedrock-fallback-enabled", "Value": "true"},
                {"Name": "/aws-hybrid-llm/test/local-timeout-ms", "Value": "25000"},
                {"Name": "/aws-hybrid-llm/test/bedrock-fallback-model",
                 "Value": "anthropic.claude-3-haiku-20240307-v1:0"},
            ]
        }
        mock_ecs.describe_services.return_value = {
            "services": [{"desiredCount": 1, "runningCount": 1}]
        }

        # Simulate Ollama timeout
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        # Mock Bedrock response
        bedrock_body = json.dumps({
            "content": [{"text": "Bedrock fallback answer"}],
            "usage": {"input_tokens": 15, "output_tokens": 25}
        }).encode()

        mock_bedrock.invoke_model.return_value = {
            "body": MagicMock(read=Mock(return_value=bedrock_body))
        }
        mock_cw.put_metric_data.return_value = {}

        import handler
        handler._config_cache = {}

        result = handler.lambda_handler(
            {"action": "infer", "query": "Complex reasoning question?"},
            {}
        )

        self.assertEqual(result["source"], "bedrock_fallback")
        self.assertTrue(result["fallback_triggered"])
        self.assertIn("Bedrock fallback answer", result["answer"])

    @patch("handler.ssm")
    @patch("handler.ecs")
    @patch("handler.urllib.request.urlopen")
    def test_cold_start_scale_up(self, mock_urlopen, mock_ecs, mock_ssm):
        """ECS at 0 tasks — scale up triggered, fallback used for this request."""
        mock_ssm.get_parameters_by_path.return_value = {"Parameters": [
            {"Name": "/aws-hybrid-llm/test/default-model", "Value": "llama3.2"},
            {"Name": "/aws-hybrid-llm/test/bedrock-fallback-enabled", "Value": "true"},
            {"Name": "/aws-hybrid-llm/test/local-timeout-ms", "Value": "25000"},
            {"Name": "/aws-hybrid-llm/test/bedrock-fallback-model",
             "Value": "anthropic.claude-3-haiku-20240307-v1:0"},
        ]}

        # ECS at desired=0
        mock_ecs.describe_services.return_value = {
            "services": [{"desiredCount": 0, "runningCount": 0}]
        }
        mock_ecs.update_service.return_value = {}

        import handler
        handler._config_cache = {}

        with patch("handler.bedrock_runtime") as mock_bedrock, \
             patch("handler.cloudwatch") as mock_cw:

            bedrock_body = json.dumps({
                "content": [{"text": "Cold start fallback answer"}],
                "usage": {"input_tokens": 10, "output_tokens": 15}
            }).encode()
            mock_bedrock.invoke_model.return_value = {
                "body": MagicMock(read=Mock(return_value=bedrock_body))
            }
            mock_cw.put_metric_data.return_value = {}

            result = handler.lambda_handler(
                {"action": "infer", "query": "Test query"},
                {}
            )

        # Scale up should have been called
        mock_ecs.update_service.assert_called_once_with(
            cluster="test-cluster",
            service="test-service",
            desiredCount=1
        )
        self.assertTrue(result["fallback_triggered"])

    @patch("handler.urllib.request.urlopen")
    def test_warmup_action(self, mock_urlopen):
        """Warmup ping hits /api/tags."""
        mock_cm = MagicMock()
        mock_cm.__enter__ = Mock(return_value=Mock())
        mock_cm.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_cm

        import handler
        result = handler.lambda_handler({"action": "warmup"}, {})
        self.assertEqual(result["status"], "warm")

    def test_missing_query_returns_400(self):
        import handler
        with patch("handler.get_config") as mock_cfg, \
             patch("handler.ensure_ecs_running") as mock_ecs:
            mock_cfg.return_value = {
                "default_model": "llama3.2",
                "fallback_enabled": True,
                "local_timeout_ms": 25000,
                "bedrock_fallback_model": "anthropic.claude-3-haiku-20240307-v1:0"
            }
            mock_ecs.return_value = True
            result = handler.lambda_handler({"action": "infer", "query": ""}, {})
        self.assertEqual(result.get("statusCode"), 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
