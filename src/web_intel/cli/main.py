"""
Main CLI application for Web Intelligence System.

Provides the primary command-line interface for:
- Crawling websites
- Querying crawled content
- Managing configuration
- Viewing system status
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.markdown import Markdown

from web_intel import __version__
from web_intel.config import load_config, Settings
from web_intel.utils.logging import setup_logging, get_logger

# Initialize Typer app
app = typer.Typer(
    name="web-intel",
    help="Website Intelligence System - Crawl, understand, and query websites",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(
            f"[bold blue]Web Intelligence System[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        dir_okay=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Web Intelligence System - Crawl and query websites intelligently.

    Use 'web-intel --help' for command list.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)


@app.command()
def crawl(
    url: str = typer.Argument(
        ...,
        help="URL to start crawling from",
    ),
    max_pages: int = typer.Option(
        100,
        "--max-pages",
        "-m",
        help="Maximum pages to crawl",
        min=1,
        max=1000,
    ),
    depth: int = typer.Option(
        3,
        "--depth",
        "-d",
        help="Maximum crawl depth",
        min=1,
        max=10,
    ),
    delay: float = typer.Option(
        1.0,
        "--delay",
        help="Delay between requests in seconds",
        min=0.1,
        max=30.0,
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help="Run browser in headless mode",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output database path",
    ),
    process: bool = typer.Option(
        True,
        "--process/--no-process",
        help="Process and index content after crawling",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """
    Crawl a website and index its content.

    Example:
        web-intel crawl https://example.com --max-pages 50
    """
    try:
        asyncio.run(_crawl_async(
            url=url,
            max_pages=max_pages,
            depth=depth,
            delay=delay,
            headless=headless,
            output=output,
            process=process,
            config_file=config_file,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Crawl cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Crawl failed")
        raise typer.Exit(1)


async def _crawl_async(
    url: str,
    max_pages: int,
    depth: int,
    delay: float,
    headless: bool,
    output: Optional[Path],
    process: bool,
    config_file: Optional[Path],
) -> None:
    """Async crawl implementation."""
    from urllib.parse import urlparse

    from web_intel.config import load_config
    from web_intel.storage import Database, PageRepository, CrawlRepository
    from web_intel.crawler import Crawler
    from web_intel.extraction import ContentExtractor
    from web_intel.understanding import TextChunker
    from web_intel.embeddings import Embedder
    from web_intel.vector_store import VectorStore
    from web_intel.storage.models import PageRecord, PageStatus, CrawlRecord, ChunkRecord

    # Load configuration
    settings = load_config(config_file)

    # Override with CLI options
    settings.crawler.max_pages = max_pages
    settings.crawler.max_depth = depth
    settings.crawler.delay_seconds = delay
    settings.browser.headless = headless

    if output:
        settings.storage.database_path = str(output)

    console.print(Panel(
        f"[bold]Crawling:[/bold] {url}\n"
        f"[dim]Max pages: {max_pages} | Depth: {depth} | Delay: {delay}s[/dim]",
        title="Web Intelligence Crawler",
        border_style="blue",
    ))

    # Initialize database
    db = Database.initialize(settings)

    # Create repositories
    crawl_repo = CrawlRepository(db)
    page_repo = PageRepository(db)

    # Create crawl record
    domain = urlparse(url).netloc
    crawl_record = CrawlRecord(
        start_url=url,
        domain=domain,
        max_pages=max_pages,
        max_depth=depth,
    )
    crawl_id = crawl_repo.create(crawl_record)

    # Initialize crawler
    crawler = Crawler(
        browser_settings=settings.browser,
        crawler_settings=settings.crawler,
    )

    extractor = ContentExtractor()

    # Initialize processing components if needed
    embedder = None
    vector_store = None
    chunker = None

    if process:
        embedder = Embedder.from_settings(settings)
        vector_store = VectorStore.from_settings(settings, db)
        chunker = TextChunker(chunk_size=1000, overlap=100)

    pages_crawled = 0
    pages_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        crawl_task = progress.add_task(
            "[cyan]Crawling...",
            total=max_pages,
        )

        async for page_content in crawler.crawl(url):
            pages_crawled += 1
            progress.update(crawl_task, advance=1)
            progress.update(
                crawl_task,
                description=f"[cyan]Crawling... ({pages_crawled}/{max_pages})",
            )

            # Extract content
            extracted = extractor.extract(
                html=page_content.html,
                url=page_content.url,
            )

            # Save page record
            page_record = PageRecord(
                crawl_id=crawl_id,
                url=page_content.url,
                title=page_content.title or extracted.title,
                content_text=extracted.main_content,
                content_html=page_content.html[:50000],  # Limit HTML storage
                word_count=len(extracted.main_content.split()),
                status=PageStatus.CRAWLED,
                depth=0,
            )
            page_id = page_repo.insert(page_record)

            # Process and index if enabled
            if process and embedder and vector_store and chunker:
                try:
                    # Chunk content
                    chunks = chunker.chunk(extracted.main_content)

                    for chunk in chunks:
                        # Generate embedding
                        embedding_result = embedder.embed(chunk.text)

                        # Save chunk and add to vector store
                        chunk_record = ChunkRecord(
                            page_id=page_id,
                            chunk_index=chunk.index,
                            text=chunk.text,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                        )
                        vector_store.add_chunk(
                            chunk_record, embedding_result.embedding)

                    pages_processed += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to process page {page_content.url}: {e}")

    # Update crawl record
    crawl_repo.update_stats(
        crawl_id, pages_crawled=pages_crawled, pages_processed=pages_processed)
    crawl_repo.complete(crawl_id)

    # Summary
    console.print()
    console.print(Panel(
        f"[green]✓ Crawl complete![/green]\n\n"
        f"Pages crawled: [bold]{pages_crawled}[/bold]\n"
        f"Pages processed: [bold]{pages_processed}[/bold]\n"
        f"Database: [dim]{settings.storage.database_path}[/dim]",
        title="Summary",
        border_style="green",
    ))


@app.command()
def query(
    question: Optional[str] = typer.Argument(
        None,
        help="Question to ask (omit for interactive mode)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Start interactive query session",
    ),
    crawl_id: Optional[int] = typer.Option(
        None,
        "--crawl",
        help="Filter to specific crawl ID",
    ),
    show_sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Show source citations",
    ),
    max_results: int = typer.Option(
        5,
        "--max-results",
        "-n",
        help="Maximum results to retrieve",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """
    Query the knowledge base.

    Ask questions about crawled website content.

    Examples:
        web-intel query "What products do you offer?"
        web-intel query --interactive
    """
    if not question and not interactive:
        interactive = True

    try:
        _query_sync(
            question=question,
            interactive=interactive,
            crawl_id=crawl_id,
            show_sources=show_sources,
            max_results=max_results,
            config_file=config_file,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Query session ended[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Query failed")
        raise typer.Exit(1)


def _query_sync(
    question: Optional[str],
    interactive: bool,
    crawl_id: Optional[int],
    show_sources: bool,
    max_results: int,
    config_file: Optional[Path],
) -> None:
    """Synchronous query implementation."""
    from web_intel.config import load_config
    from web_intel.storage import Database
    from web_intel.vector_store import VectorStore
    from web_intel.memory_store import MemoryStore
    from web_intel.query_executor import QueryExecutor
    from web_intel.answer_generator import AnswerGenerator

    # Load configuration
    settings = load_config(config_file)

    # Initialize components
    db = Database.initialize(settings)
    vector_store = VectorStore.from_settings(settings, db)
    memory_store = MemoryStore.from_settings(settings, db)

    # Initialize answer generator
    answer_gen = None
    if settings.local_llm.enabled:
        from web_intel.llm import LocalLLM
        llm = LocalLLM.from_settings(settings)
        answer_generator = AnswerGenerator(llm=llm)
        answer_gen = answer_generator.as_callable()

    # Create executor
    executor = QueryExecutor(
        database=db,
        vector_store=vector_store,
        memory_store=memory_store,
        answer_generator=answer_gen,
        top_k=max_results,
    )

    if interactive:
        _interactive_query(executor, crawl_id, show_sources)
    else:
        _single_query(executor, question, crawl_id, show_sources)


def _single_query(
    executor,
    question: str,
    crawl_id: Optional[int],
    show_sources: bool,
) -> None:
    """Execute a single query."""
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with console.status("[cyan]Thinking..."):
        result = executor.execute(
            query=question,
            crawl_id=crawl_id,
        )

    # Display answer
    console.print(Panel(
        Markdown(result.answer),
        title="Answer",
        border_style="green",
    ))

    # Display sources
    if show_sources and result.sources:
        console.print()
        table = Table(title="Sources", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan")
        table.add_column("URL", style="dim")
        table.add_column("Relevance", justify="right")

        for i, source in enumerate(result.sources[:5], 1):
            table.add_row(
                str(i),
                source.title[:50] +
                "..." if len(source.title) > 50 else source.title,
                source.url[:60] +
                "..." if len(source.url) > 60 else source.url,
                f"{source.relevance_score:.2f}",
            )

        console.print(table)


def _interactive_query(
    executor,
    crawl_id: Optional[int],
    show_sources: bool,
) -> None:
    """Run interactive query session."""
    console.print(Panel(
        "[bold]Interactive Query Mode[/bold]\n\n"
        "Ask questions about the crawled content.\n"
        "Type [cyan]'quit'[/cyan] or [cyan]'exit'[/cyan] to end the session.\n"
        "Type [cyan]'help'[/cyan] for more commands.",
        border_style="blue",
    ))

    session_id = executor.create_session()

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                break

            if question.lower() == "help":
                console.print(
                    "\n[bold]Commands:[/bold]\n"
                    "  quit/exit - End session\n"
                    "  help - Show this help\n"
                    "  sources on/off - Toggle source display\n"
                    "  clear - Clear conversation history\n"
                )
                continue

            if question.lower() == "sources on":
                show_sources = True
                console.print("[dim]Sources display enabled[/dim]")
                continue

            if question.lower() == "sources off":
                show_sources = False
                console.print("[dim]Sources display disabled[/dim]")
                continue

            if question.lower() == "clear":
                session_id = executor.create_session()
                console.print("[dim]Conversation cleared[/dim]")
                continue

            # Execute query
            with console.status("[cyan]Thinking..."):
                result = executor.execute(
                    query=question,
                    session_id=session_id,
                    crawl_id=crawl_id,
                )

            # Display answer
            console.print(
                f"\n[bold green]Assistant:[/bold green] {result.answer}")

            # Display sources inline
            if show_sources and result.sources:
                sources_text = ", ".join(
                    f"[{s.title[:30]}]" for s in result.sources[:3]
                )
                console.print(f"[dim]Sources: {sources_text}[/dim]")

        except EOFError:
            break

    console.print("\n[dim]Session ended[/dim]")


@app.command()
def status(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """
    Show system and crawl status.

    Displays database statistics, crawl history, and system health.
    """
    try:
        _show_status(config_file)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Status check failed")
        raise typer.Exit(1)


def _show_status(config_file: Optional[Path]) -> None:
    """Display system status."""
    from web_intel.config import load_config
    from web_intel.storage import Database, CrawlRepository, PageRepository, ChunkRepository

    settings = load_config(config_file)

    # Check database
    db_path = Path(settings.storage.database_path)
    db_exists = db_path.exists()

    console.print(Panel(
        f"[bold]Web Intelligence System[/bold] v{__version__}",
        border_style="blue",
    ))

    # Database status
    console.print("\n[bold]Database:[/bold]")
    if db_exists:
        db = Database.initialize(settings)

        # Get statistics
        crawl_repo = CrawlRepository(db)
        chunk_repo = ChunkRepository(db)

        crawls = crawl_repo.get_all(limit=1000)
        crawl_count = len(crawls)

        # Count pages and chunks directly
        page_result = db.fetch_one("SELECT COUNT(*) as count FROM pages")
        page_count = page_result["count"] if page_result else 0
        chunk_count = chunk_repo.count()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Path", str(db_path))
        table.add_row("Size", f"{db_path.stat().st_size / 1024 / 1024:.1f} MB")
        table.add_row("Crawls", str(crawl_count))
        table.add_row("Pages", str(page_count))
        table.add_row("Chunks", str(chunk_count))

        console.print(table)

        # Recent crawls
        if crawl_count > 0:
            console.print("\n[bold]Recent Crawls:[/bold]")
            recent_crawls = crawls[:5]

            crawl_table = Table(show_header=True)
            crawl_table.add_column("ID", style="dim", width=4)
            crawl_table.add_column("URL", style="cyan")
            crawl_table.add_column("Pages", justify="right")
            crawl_table.add_column("Status")
            crawl_table.add_column("Date", style="dim")

            for crawl in recent_crawls:
                status_style = "green" if crawl.status.value == "completed" else "yellow"
                crawl_table.add_row(
                    str(crawl.id),
                    crawl.start_url[:40] +
                    "..." if len(crawl.start_url) > 40 else crawl.start_url,
                    str(crawl.pages_crawled),
                    f"[{status_style}]{crawl.status.value}[/{status_style}]",
                    crawl.started_at.strftime(
                        "%Y-%m-%d %H:%M") if crawl.started_at else "N/A",
                )

            console.print(crawl_table)
    else:
        console.print(f"  [yellow]Database not found:[/yellow] {db_path}")
        console.print("  [dim]Run 'web-intel crawl <url>' to create it[/dim]")

    # Configuration
    console.print("\n[bold]Configuration:[/bold]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value")

    config_table.add_row(
        "LLM (Local)", "✓ Enabled" if settings.local_llm.enabled else "✗ Disabled")
    config_table.add_row("LLM Model", settings.local_llm.model_name)
    config_table.add_row("Embedding Model", settings.embedding.model_name)
    config_table.add_row("Max Pages", str(settings.crawler.max_pages))

    console.print(config_table)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Show current configuration",
    ),
    init: bool = typer.Option(
        False,
        "--init",
        help="Create default configuration file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for config file",
    ),
) -> None:
    """
    Configuration management.

    View or initialize configuration files.

    Examples:
        web-intel config --show
        web-intel config --init --output ./my-config.yaml
    """
    if init:
        _init_config(output)
    elif show:
        _show_config()
    else:
        console.print(
            "Use --show to view config or --init to create default config")


def _show_config() -> None:
    """Show current configuration."""
    from web_intel.config import load_config

    settings = load_config()

    # Convert to dict and display
    config_dict = settings.model_dump()

    console.print(Panel(
        "[bold]Current Configuration[/bold]",
        border_style="blue",
    ))

    for section, values in config_dict.items():
        console.print(f"\n[bold cyan]{section}:[/bold cyan]")
        if isinstance(values, dict):
            for key, value in values.items():
                console.print(f"  {key}: [dim]{value}[/dim]")
        else:
            console.print(f"  {values}")


def _init_config(output: Optional[Path]) -> None:
    """Create default configuration file."""
    import yaml
    from web_intel.config import Settings

    default_settings = Settings()
    config_dict = default_settings.model_dump()

    output_path = output or Path("config.yaml")

    if output_path.exists():
        if not typer.confirm(f"File {output_path} exists. Overwrite?"):
            raise typer.Exit(0)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Configuration saved to: {output_path}")


# Add subcommands for specific operations
@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum results",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """
    Search indexed pages (without LLM).

    Fast keyword and vector search without answer generation.

    Example:
        web-intel search "pricing plans"
    """
    from web_intel.config import load_config
    from web_intel.storage import Database
    from web_intel.vector_store import VectorStore

    settings = load_config(config_file)
    db = Database.initialize(settings)
    vector_store = VectorStore.from_settings(settings, db)

    # Load vector index
    vector_store.load_index()

    console.print(f"\n[bold]Searching:[/bold] {query}\n")

    # Search using text (embedder is inside vector_store)
    with console.status("[cyan]Searching..."):
        results = vector_store.search_text(
            query=query,
            top_k=limit,
        )

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display results
    table = Table(
        title=f"Search Results ({len(results)} found)", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Content", style="white")

    for i, result in enumerate(results, 1):
        # Truncate content
        content = result.text[:200] + \
            "..." if len(result.text) > 200 else result.text
        content = content.replace("\n", " ")

        table.add_row(
            str(i),
            f"{result.score:.3f}",
            content,
        )

    console.print(table)


if __name__ == "__main__":
    app()
