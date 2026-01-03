"""Unit tests for the neural_network.protobuf.compile_protobuf module."""

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from neural_network.protobuf.compile_protobuf import OUT_DIR, PROJECT_ROOT, PROTO_DIR, compile_protobuf, main


@pytest.fixture
def mock_logger() -> Generator[MagicMock]:
    """Fixture for patching the logger used in compile_protobuf module."""
    with patch("neural_network.protobuf.compile_protobuf.logger") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_mkdir() -> Generator[MagicMock]:
    """Fixture for patching Path.mkdir used in compile_protobuf module."""
    with patch("neural_network.protobuf.compile_protobuf.Path.mkdir") as mock:
        mock.return_value = None
        yield mock


@pytest.fixture(autouse=True)
def mock_glob() -> Generator[MagicMock]:
    """Fixture for patching Path.glob used in compile_protobuf module."""
    with patch("neural_network.protobuf.compile_protobuf.Path.glob") as mock:
        mock.return_value = []
        yield mock


@pytest.fixture
def mock_subprocess_run() -> Generator[MagicMock]:
    """Fixture for patching subprocess.run used in compile_protobuf module."""
    with patch("neural_network.protobuf.compile_protobuf.subprocess.run") as mock:
        yield mock


@pytest.fixture
def mock_compile_protobuf() -> Generator[MagicMock]:
    """Fixture for patching compile_protobuf function itself."""
    with patch("neural_network.protobuf.compile_protobuf.compile_protobuf") as mock:
        yield mock


@pytest.fixture
def mock_sys_exit() -> Generator[MagicMock]:
    """Fixture for patching sys.exit used in compile_protobuf module."""
    with patch("neural_network.protobuf.compile_protobuf.sys.exit") as mock:
        yield mock


class TestCompileProtobuf:
    """Test cases for Protobuf compilation and CLI entry point."""

    def test_compile_protobuf_success(
        self, mock_logger: MagicMock, mock_glob: MagicMock, mock_mkdir: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Test successful compilation of Protobuf files."""
        mock_proto_files = [Path("test1.proto"), Path("test2.proto")]
        mock_glob.return_value = mock_proto_files

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        result = compile_protobuf()

        assert result is True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_glob.assert_called_once_with("*.proto")

        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        cmd = args[0]
        assert sys.executable in cmd
        assert "grpc_tools.protoc" in cmd
        assert kwargs["cwd"] == PROJECT_ROOT
        assert kwargs["check"] is False

        mock_logger.info.assert_has_calls(
            [
                call("Generating Protobuf files..."),
                call("Protobuf directory: %s", PROTO_DIR),
                call("Output directory: %s", OUT_DIR),
                call("Protobuf generation complete!"),
                call("Generated files for: %s", [f.name for f in mock_proto_files]),
            ]
        )

    def test_compile_protobuf_no_proto_files(
        self, mock_logger: MagicMock, mock_glob: MagicMock, mock_mkdir: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Test behavior when no .proto files are found."""
        mock_glob.return_value = []

        result = compile_protobuf()

        assert result is False
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_glob.assert_called_once_with("*.proto")

        mock_subprocess_run.assert_not_called()

        mock_logger.warning.assert_called_once_with("No .proto files found in the Protobuf directory.")

    def test_compile_protobuf_subprocess_error(
        self, mock_logger: MagicMock, mock_glob: MagicMock, mock_mkdir: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Test behavior when subprocess returns an error during compilation."""
        mock_glob.return_value = [Path("test1.proto")]

        error_message = "Error: something went wrong"
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = error_message
        mock_subprocess_run.return_value = mock_process

        result = compile_protobuf()

        assert result is False
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_glob.assert_called_once_with("*.proto")

        mock_subprocess_run.assert_called_once()

        mock_logger.info.assert_has_calls(
            [
                call("Generating Protobuf files..."),
                call("Protobuf directory: %s", PROTO_DIR),
                call("Output directory: %s", OUT_DIR),
            ]
        )
        mock_logger.error.assert_has_calls(
            [
                call("Error during Protobuf compilation:"),
                call(error_message),
            ]
        )

    def test_compile_protobuf_exception(
        self, mock_logger: MagicMock, mock_glob: MagicMock, mock_mkdir: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Test behavior when an exception is raised during compilation."""
        mock_glob.return_value = [Path("test1.proto")]

        mock_subprocess_run.side_effect = Exception("Unexpected error")

        result = compile_protobuf()

        assert result is False
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_glob.assert_called_once_with("*.proto")
        mock_subprocess_run.assert_called_once()

        mock_logger.info.assert_has_calls(
            [
                call("Generating Protobuf files..."),
                call("Protobuf directory: %s", PROTO_DIR),
                call("Output directory: %s", OUT_DIR),
            ]
        )
        mock_logger.exception.assert_called_once_with("Error during Protobuf compilation")

    def test_main_success(self, mock_compile_protobuf: MagicMock, mock_sys_exit: MagicMock) -> None:
        """Test main function exits with code 0 on success."""
        mock_compile_protobuf.return_value = True

        main()

        mock_compile_protobuf.assert_called_once()
        mock_sys_exit.assert_called_once_with(0)

    def test_main_failure(self, mock_compile_protobuf: MagicMock, mock_sys_exit: MagicMock) -> None:
        """Test main function exits with code 1 on failure."""
        mock_compile_protobuf.return_value = False

        main()

        mock_compile_protobuf.assert_called_once()
        mock_sys_exit.assert_called_once_with(1)
