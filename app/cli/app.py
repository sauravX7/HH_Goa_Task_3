"""
app/cli/app.py - Typer CLI application for the Verification Pipeline.
Supports --image, --demo, --threshold, --network options and standalone subcommands.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from app.blockchain.client import BlockchainClient
from app.config import config
from app.orchestrator.context import PipelineContext
from app.orchestrator.pipeline import PipelineOrchestrator

cli_app = typer.Typer(
    name="face-provenance",
    help="🛡️ Automated Face Provenance & Blockchain Verification Pipeline CLI",
    add_completion=False,
)
console = Console()


@cli_app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    image: Optional[Path] = typer.Option(
        None,
        "--image",
        "-i",
        help="Path to input face image file (JPEG/PNG).",
        exists=False,
        dir_okay=False,
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        "-d",
        help="Enable hackathon demo mode with paced stage transitions and snapshot generation.",
    ),
    demo_delay: float = typer.Option(
        1.5,
        "--demo-delay",
        help="Transition delay in seconds between stages in demo mode.",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Cosine similarity threshold for match validation (default: 0.60).",
    ),
    network: Optional[str] = typer.Option(
        None,
        "--network",
        "-n",
        help="EVM target network: 'hardhat', 'anvil', or 'polygon_amoy'.",
    ),
    contract: Optional[str] = typer.Option(
        None,
        "--contract",
        "-c",
        help="Deployed smart contract address (hex).",
    ),
    artifacts_dir: Optional[Path] = typer.Option(
        None,
        "--artifacts-dir",
        "-a",
        help="Directory to save generated artifacts (default: artifacts/).",
    ),
):
    """
    Execute the 10-stage face verification pipeline.
    """
    if ctx.invoked_subcommand is not None:
        return

    if image is None:
        # Default test face image fallback if none provided
        default_asset = Path(__file__).resolve().parent.parent.parent / "tests" / "assets" / "test_face.jpg"
        if default_asset.exists():
            image = default_asset
        else:
            console.print("[bold red]Error:[/bold red] Missing option '--image' / '-i'. Please provide an input image path.")
            raise typer.Exit(code=1)

    image_path = Path(image)
    if not image_path.exists():
        console.print(f"[bold red]Error:[/bold red] Specified image does not exist: {image_path}")
        raise typer.Exit(code=1)

    effective_threshold = threshold if threshold is not None else config.effective_similarity_threshold
    effective_network = network or config.effective_network

    pipe_ctx = PipelineContext(
        image_path=image_path,
        artifacts_dir=artifacts_dir or config.paths.artifacts_dir,
        is_demo=demo,
        similarity_threshold=effective_threshold,
        network=effective_network,
        contract_address=contract,
    )

    orchestrator = PipelineOrchestrator()
    orchestrator.demo_recorder.step_delay = demo_delay

    success = orchestrator.run(pipe_ctx)
    if not success:
        console.print(f"\n[bold yellow]Pipeline finished with status: ABORTED / FAILED[/bold yellow]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]✓ Pipeline completed successfully.[/bold green]")


@cli_app.command("verify")
def verify_command(
    canonical_file: Path = typer.Option(
        config.paths.canonical_post_file,
        "--canonical",
        "-c",
        help="Path to canonical_post.json to verify against on-chain records.",
    ),
    network: Optional[str] = typer.Option(
        None,
        "--network",
        "-n",
        help="EVM target network: 'hardhat', 'anvil', or 'polygon_amoy'.",
    ),
    contract: Optional[str] = typer.Option(
        None,
        "--contract",
        help="Contract address override.",
    ),
):
    """
    Verify local canonical post against smart contract on-chain records.
    """
    from app.verification.engine import BlockchainVerifier
    if not canonical_file.exists():
        console.print(f"[bold red]Error:[/bold red] Canonical file does not exist: {canonical_file}")
        raise typer.Exit(code=1)

    client = BlockchainClient(network=network, contract_address=contract)
    verifier = BlockchainVerifier(client)
    res = verifier.verify_canonical_data(canonical_file)

    console.print(f"\n[bold cyan]Verification Outcome:[/bold cyan] [bold green]{res.verification_status.value}[/bold green]")
    console.print(f"Computed Hash: {res.computed_content_hash}")
    console.print(f"On-Chain Hash: {res.on_chain_content_hash}")
    console.print(f"Hashes Match: {res.hashes_match}")
    console.print(f"Rationale: {res.rationale}")

    if not res.is_verified:
        raise typer.Exit(code=1)


@cli_app.command("tamper")
def tamper_command(
    canonical_file: Path = typer.Option(
        config.paths.canonical_post_file,
        "--canonical",
        "-c",
        help="Path to canonical_post.json baseline.",
    ),
    network: Optional[str] = typer.Option(
        None,
        "--network",
        "-n",
        help="EVM target network.",
    ),
    contract: Optional[str] = typer.Option(
        None,
        "--contract",
        help="Contract address override.",
    ),
):
    """
    Run 5-scenario automated tamper attack demonstration against on-chain records.
    """
    from app.tamper.engine import TamperDetector
    import json

    if not canonical_file.exists():
        console.print(f"[bold red]Error:[/bold red] Canonical file does not exist: {canonical_file}")
        raise typer.Exit(code=1)

    canonical_data = json.loads(canonical_file.read_text(encoding="utf-8"))
    client = BlockchainClient(network=network, contract_address=contract)
    detector = TamperDetector(client)
    res = detector.run_5_tamper_scenarios(canonical_data)

    from app.cli.ui import ConsoleUI
    ui = ConsoleUI()
    ui.render_tamper_matrix(res.model_dump(mode="json"))


@cli_app.command("deploy")
def deploy_command(
    network: Optional[str] = typer.Option(
        None,
        "--network",
        "-n",
        help="EVM target network.",
    ),
):
    """
    Deploy the FaceProvenanceRegistry smart contract to the configured network.
    """
    client = BlockchainClient(network=network)
    try:
        addr, receipt = client.deploy_contract()
        console.print(f"[bold green]✓ Smart contract deployed successfully![/bold green]")
        console.print(f"Contract Address: [bold cyan]{addr}[/bold cyan]")
        console.print(f"Transaction Hash: {receipt.get('transactionHash')}")
        console.print(f"Block Number: {receipt.get('blockNumber')}")
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")
        raise typer.Exit(code=1)
