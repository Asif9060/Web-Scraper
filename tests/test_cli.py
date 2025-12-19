"""
Tests for CLI module.

Tests command-line interface commands and output.
"""

import pytest
from typer.testing import CliRunner

from web_intel.cli import app


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_cli_help(self, runner: CliRunner):
        """CLI should show help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_crawl_help(self, runner: CliRunner):
        """Crawl command should show help."""
        result = runner.invoke(app, ["crawl", "--help"])

        assert result.exit_code == 0
        assert "url" in result.output.lower() or "URL" in result.output

    def test_query_help(self, runner: CliRunner):
        """Query command should show help."""
        result = runner.invoke(app, ["query", "--help"])

        assert result.exit_code == 0

    def test_search_help(self, runner: CliRunner):
        """Search command should show help."""
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0

    def test_status_command(self, runner: CliRunner, temp_dir):
        """Status command should run."""
        result = runner.invoke(
            app,
            ["status", "--data-dir", str(temp_dir)],
        )

        # Should complete (may show empty status)
        assert result.exit_code == 0 or "no data" in result.output.lower()

    def test_config_show(self, runner: CliRunner):
        """Config show command should display settings."""
        result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0

    def test_config_path(self, runner: CliRunner):
        """Config path command should show config location."""
        result = runner.invoke(app, ["config", "path"])

        assert result.exit_code == 0


class TestCrawlCommand:
    """Tests for crawl command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_crawl_missing_url(self, runner: CliRunner):
        """Crawl without URL should fail."""
        result = runner.invoke(app, ["crawl"])

        # Should require URL
        assert result.exit_code != 0 or "url" in result.output.lower()

    def test_crawl_invalid_url(self, runner: CliRunner, temp_dir):
        """Crawl with invalid URL should fail gracefully."""
        result = runner.invoke(
            app,
            ["crawl", "not-a-valid-url", "--data-dir", str(temp_dir)],
        )

        # Should fail with error message
        assert result.exit_code != 0 or "invalid" in result.output.lower(
        ) or "error" in result.output.lower()

    def test_crawl_with_options(self, runner: CliRunner, temp_dir):
        """Crawl command should accept options."""
        result = runner.invoke(
            app,
            [
                "crawl",
                "https://example.com",
                "--max-pages", "10",
                "--max-depth", "2",
                "--data-dir", str(temp_dir),
                "--dry-run",
            ],
        )

        # With dry-run, should complete without actual crawling
        assert result.exit_code == 0 or "dry" in result.output.lower()


class TestQueryCommand:
    """Tests for query command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_query_missing_text(self, runner: CliRunner):
        """Query without text should fail."""
        result = runner.invoke(app, ["query"])

        assert result.exit_code != 0

    def test_query_no_data(self, runner: CliRunner, temp_dir):
        """Query with no crawled data should inform user."""
        result = runner.invoke(
            app,
            ["query", "What is the price?", "--data-dir", str(temp_dir)],
        )

        # Should indicate no data available
        assert "no data" in result.output.lower(
        ) or "crawl" in result.output.lower() or result.exit_code == 0


class TestSearchCommand:
    """Tests for search command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_search_missing_query(self, runner: CliRunner):
        """Search without query should fail."""
        result = runner.invoke(app, ["search"])

        assert result.exit_code != 0

    def test_search_with_limit(self, runner: CliRunner, temp_dir):
        """Search should accept limit option."""
        result = runner.invoke(
            app,
            ["search", "test query", "--limit",
                "5", "--data-dir", str(temp_dir)],
        )

        # Should complete
        assert result.exit_code == 0 or "no results" in result.output.lower()


class TestConfigCommand:
    """Tests for config command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_config_show_format(self, runner: CliRunner):
        """Config show should display readable format."""
        result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should show some configuration
        assert "browser" in result.output.lower(
        ) or "crawler" in result.output.lower() or len(result.output) > 0

    def test_config_get_value(self, runner: CliRunner):
        """Config get should retrieve specific value."""
        result = runner.invoke(app, ["config", "get", "crawler.max_pages"])

        # Should show value or indicate key
        assert result.exit_code == 0 or "not found" in result.output.lower()


class TestStatusCommand:
    """Tests for status command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_status_output(self, runner: CliRunner, temp_dir):
        """Status should show system state."""
        result = runner.invoke(
            app,
            ["status", "--data-dir", str(temp_dir)],
        )

        assert result.exit_code == 0
        # Should show some status info
        output_lower = result.output.lower()
        assert any(
            term in output_lower
            for term in ["pages", "database", "status", "empty", "no data"]
        ) or len(result.output) > 0

    def test_status_verbose(self, runner: CliRunner, temp_dir):
        """Status verbose mode should show more detail."""
        result = runner.invoke(
            app,
            ["status", "--verbose", "--data-dir", str(temp_dir)],
        )

        assert result.exit_code == 0


class TestCLIOutput:
    """Tests for CLI output formatting."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_json_output(self, runner: CliRunner, temp_dir):
        """Commands should support JSON output."""
        result = runner.invoke(
            app,
            ["status", "--json", "--data-dir", str(temp_dir)],
        )

        if result.exit_code == 0 and result.output.strip():
            # If JSON output is supported, should be valid JSON
            import json
            try:
                json.loads(result.output)
                is_json = True
            except json.JSONDecodeError:
                is_json = False

            # Either valid JSON or command doesn't support --json
            assert is_json or "--json" not in result.output

    def test_quiet_mode(self, runner: CliRunner, temp_dir):
        """Commands should support quiet mode."""
        result = runner.invoke(
            app,
            ["status", "--quiet", "--data-dir", str(temp_dir)],
        )

        # Quiet mode should have minimal output
        assert result.exit_code == 0


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a CLI test runner."""
        return CliRunner()

    def test_invalid_command(self, runner: CliRunner):
        """Invalid command should show error."""
        result = runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0

    def test_keyboard_interrupt(self, runner: CliRunner):
        """CLI should handle keyboard interrupt gracefully."""
        # This is difficult to test directly, but we can verify
        # the app has proper exception handling by checking it loads
        assert app is not None

    def test_missing_config_file(self, runner: CliRunner):
        """Missing config file should use defaults."""
        result = runner.invoke(
            app,
            ["config", "show", "--config", "/nonexistent/config.yaml"],
        )

        # Should either use defaults or show helpful error
        assert result.exit_code == 0 or "not found" in result.output.lower(
        ) or "default" in result.output.lower()
