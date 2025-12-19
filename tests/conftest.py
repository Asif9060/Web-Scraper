"""
Shared pytest fixtures for Web Intelligence System tests.

Provides reusable fixtures for:
- Configuration and settings
- Database instances
- Sample data
- Temporary resources
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from web_intel.config import Settings
from web_intel.storage import Database, reset_database


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Reset global database state before and after each test.

    This ensures tests are isolated and don't share global state.
    """
    reset_database()
    yield
    reset_database()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """
    Provide test settings with temporary database path.

    Uses minimal resource settings for fast tests.
    """
    return Settings(
        storage={"database_path": str(temp_dir / "test.db")},
        local_llm={"enabled": False},  # Disable LLM for unit tests
        embedding={"batch_size": 8},
    )


@pytest.fixture
def database(test_settings: Settings) -> Generator[Database, None, None]:
    """
    Provide an initialized test database.

    Uses Database.create() for proper dependency injection.
    Creates a fresh database for each test and cleans up after.
    """
    db = Database.create(test_settings)
    yield db
    db.close()


@pytest.fixture
def sample_html() -> str:
    """Provide sample HTML for extraction tests."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="description" content="Test page description">
        <title>Test Page Title</title>
    </head>
    <body>
        <header>
            <nav>
                <a href="/home">Home</a>
                <a href="/products">Products</a>
                <a href="/about">About Us</a>
            </nav>
        </header>
        <main>
            <article>
                <h1>Welcome to Our Website</h1>
                <p>This is the main content of our test page. It contains
                important information about our products and services.</p>
                <p>We offer a wide range of products including software,
                hardware, and consulting services.</p>
                <h2>Our Products</h2>
                <ul>
                    <li>Product A - Enterprise Solution</li>
                    <li>Product B - Small Business Tool</li>
                    <li>Product C - Personal Edition</li>
                </ul>
                <h2>Contact Information</h2>
                <p>Email: contact@example.com</p>
                <p>Phone: 555-0123</p>
            </article>
        </main>
        <footer>
            <p>&copy; 2024 Test Company</p>
        </footer>
    </body>
    </html>
    """


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for chunking and processing tests."""
    return """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses on
    developing systems that can learn from data. Unlike traditional programming
    where rules are explicitly coded, machine learning algorithms discover
    patterns in data and make decisions with minimal human intervention.

    Types of Machine Learning

    There are three main types of machine learning: supervised learning,
    unsupervised learning, and reinforcement learning. Each type has its own
    strengths and is suited for different kinds of problems.

    Supervised Learning

    In supervised learning, the algorithm learns from labeled training data.
    The model makes predictions based on input features and is corrected when
    its predictions are wrong. Common applications include spam detection,
    image classification, and price prediction.

    Unsupervised Learning

    Unsupervised learning deals with unlabeled data. The algorithm tries to
    find hidden patterns or structures in the data without explicit guidance.
    Clustering and dimensionality reduction are common unsupervised techniques.

    Reinforcement Learning

    Reinforcement learning involves an agent learning to make decisions by
    taking actions in an environment to maximize cumulative reward. This
    approach is popular in robotics, game playing, and autonomous systems.

    Conclusion

    Machine learning continues to advance rapidly, with new algorithms and
    applications emerging regularly. Understanding the fundamentals is
    essential for anyone working in technology today.
    """


@pytest.fixture
def sample_urls() -> list[str]:
    """Provide sample URLs for crawler tests."""
    return [
        "https://example.com/",
        "https://example.com/products",
        "https://example.com/about",
        "https://example.com/contact",
        "https://example.com/products/item-1",
        "https://example.com/products/item-2",
        "https://example.com/blog",
        "https://example.com/blog/post-1",
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Provide sample embeddings for vector store tests."""
    np.random.seed(42)  # Reproducible
    return np.random.randn(10, 384).astype(np.float32)


@pytest.fixture
def sample_queries() -> list[dict]:
    """Provide sample queries with expected classifications."""
    return [
        {
            "query": "What products do you offer?",
            "expected_type": "list",
            "expected_terms": ["products", "offer"],
        },
        {
            "query": "How do I install the software?",
            "expected_type": "procedural",
            "expected_terms": ["install", "software"],
        },
        {
            "query": "Why is my application not working?",
            "expected_type": "explanation",
            "expected_terms": ["application", "working"],
        },
        {
            "query": "What is machine learning?",
            "expected_type": "definition",
            "expected_terms": ["machine", "learning"],
        },
        {
            "query": "Compare product A and product B",
            "expected_type": "comparison",
            "expected_terms": ["product", "compare"],
        },
        {
            "query": "Is the service available 24/7?",
            "expected_type": "yes_no",
            "expected_terms": ["service", "available"],
        },
    ]
