"""Test validation rules for public GitHub repository URLs."""

from app.repository_service import validate_repository_url


def test_validate_repository_url_accepts_https_developer_repo():
    """Allow a normal public GitHub repository URL."""
    url = "https://github.com/microsoft/vscode"
    assert validate_repository_url(url) == url


def test_validate_repository_url_rejects_non_https_or_non_github_url():
    """Reject URLs using an unsupported scheme or a non-GitHub host."""
    try:
        validate_repository_url("http://gitlab.com/org/project")
        assert False, "Expected validation error for unsupported host"
    except ValueError:
        pass

    try:
        validate_repository_url("https://example.com/owner/repo")
        assert False, "Expected validation error for unsupported host"
    except ValueError:
        pass
