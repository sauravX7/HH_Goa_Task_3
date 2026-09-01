"""
app/cli/ui.py - Rich visual console interface for CLI and demo execution.
Provides stylized stage banners, metadata tables, blockchain transaction panels,
verification badges, and tamper diff matrices.
"""

from typing import Any, Dict, List, Optional, Tuple
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class ConsoleUI:
    """Rich console visual presentation manager for the pipeline."""

    def __init__(self, console_instance: Optional[Console] = None):
        self.console = console_instance or console

    def render_header(self, image_path: str, network: str, is_demo: bool, threshold: float) -> None:
        """Renders the main pipeline application banner."""
        title = Text()
        title.append("🛡️  AUTOMATED FACE PROVENANCE & BLOCKCHAIN VERIFICATION PIPELINE\n", style="bold cyan")
        title.append("Genuine Reverse Visual Search • EVM Blockchain Registration • Deterministic Tamper Audit", style="dim white")

        meta_table = Table(box=box.SIMPLE_HEAD, show_header=False, expand=True)
        meta_table.add_column("Key", style="bold yellow", width=22)
        meta_table.add_column("Value", style="bold white")

        meta_table.add_row("Input Face Image", str(image_path))
        meta_table.add_row("Blockchain Network", f"[cyan]{network}[/cyan]")
        meta_table.add_row("Similarity Threshold", f"[green]{threshold:.2f}[/green]")
        meta_table.add_row("Execution Mode", "[bold magenta]Demo Recording Mode (--demo)[/bold magenta]" if is_demo else "[green]Production Execution[/green]")

        self.console.print(Panel(meta_table, title=title, border_style="cyan", padding=(1, 2)))

    def render_stage_banner(self, stage_number: int, stage_name: str, description: str) -> None:
        """Renders a numbered stage transition banner."""
        stage_text = Text()
        stage_text.append(f"STAGE [{stage_number:02d}/10] ", style="bold black on cyan")
        stage_text.append(f" {stage_name.upper()}\n", style="bold bright_white")
        stage_text.append(f"▶ {description}", style="dim cyan")

        self.console.print()
        self.console.print(Panel(stage_text, border_style="cyan", box=box.ROUNDED, padding=(0, 1)))

    def render_face_detection_panel(self, bbox: Optional[Tuple[int, int, int, int]], embedding_dim: int, crop_path: str) -> None:
        """Renders face detection outcome table."""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="bold yellow", width=24)
        table.add_column("Value", style="bold green")

        table.add_row("Face Bounding Box", str(bbox) if bbox else "Full Image Fallback")
        table.add_row("Feature Embedding Vector", f"{embedding_dim}-dimensional normalized float unit vector")
        table.add_row("Normalized Face Crop", str(crop_path))

        self.console.print(Panel(table, title="[bold green]✓ Stage 1: Face Detection Summary[/bold green]", border_style="green"))

    def render_search_provenance_panel(self, provider: str, query_hash: str, candidates_count: int, top_candidate: Optional[Dict[str, Any]] = None) -> None:
        """Renders reverse search provenance table."""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Property", style="bold yellow", width=24)
        table.add_column("Value", style="bold white")

        table.add_row("Active Provider", f"[bold cyan]{provider}[/bold cyan]")
        table.add_row("Query Face Crop Hash", query_hash)
        table.add_row("Candidates Discovered", f"[bold green]{candidates_count}[/bold green] candidate web posts")

        if top_candidate:
            table.add_row("Top Match Title", str(top_candidate.get("title", "N/A"))[:60])
            table.add_row("Source URL", f"[link={top_candidate.get('source_url')}]{top_candidate.get('source_url')}[/link]")
            table.add_row("Author / Handle", str(top_candidate.get("author", "N/A")))

        self.console.print(Panel(table, title="[bold green]✓ Stage 2: Reverse Visual Search Results[/bold green]", border_style="green"))

    def render_validation_match_panel(self, similarity: float, distance: float, threshold: float, is_match: bool) -> None:
        """Renders match validation outcome panel."""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="bold yellow", width=24)
        table.add_column("Value", style="bold white")

        sim_style = "bold green" if is_match else "bold red"
        table.add_row("Cosine Similarity", f"[{sim_style}]{similarity:.4f}[/{sim_style}] (Min Threshold: {threshold:.2f})")
        table.add_row("Euclidean Distance", f"[cyan]{distance:.4f}[/cyan]")
        table.add_row("Validation Status", f"[{sim_style}]{'MATCH CONFIRMED (AUTHENTIC)' if is_match else 'REJECTED (BELOW THRESHOLD)'}[/{sim_style}]")

        self.console.print(Panel(table, title="[bold green]✓ Stage 3: Match Validation Engine[/bold green]", border_style="green" if is_match else "red"))

    def render_blockchain_tx_panel(self, tx_receipt: Dict[str, Any]) -> None:
        """Renders on-chain registration transaction panel with decoded events."""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Field", style="bold yellow", width=24)
        table.add_column("Value", style="bold white")

        table.add_row("Network", f"[cyan]{tx_receipt.get('networkName', 'EVM')}[/cyan] (Chain ID: {tx_receipt.get('chainId', 'N/A')})")
        table.add_row("Smart Contract", f"[bold cyan]{tx_receipt.get('contractAddress', 'N/A')}[/bold cyan]")
        table.add_row("Transaction Hash", f"[bold green]{tx_receipt.get('transactionHash', 'N/A')}[/bold green]")
        table.add_row("Block Number", f"[yellow]{tx_receipt.get('blockNumber', 'N/A')}[/yellow]")
        table.add_row("Gas Used", f"{tx_receipt.get('gasUsed', 'N/A'):,} units")
        table.add_row("Stored Content Hash", f"[bold magenta]{tx_receipt.get('storedContentHash', 'N/A')}[/bold magenta]")

        events = tx_receipt.get("decodedEvents", [])
        if events:
            event_strs = [f"[bold]{ev.get('name', 'Event')}[/bold]" for ev in events]
            table.add_row("Decoded Events", ", ".join(event_strs))

        self.console.print(Panel(table, title="[bold green]✓ Stage 7: Blockchain Registration Receipt[/bold green]", border_style="green"))

    def render_verification_badge(self, is_verified: bool, content_hash: str, block_timestamp: int) -> None:
        """Renders bold verification badge."""
        badge_text = Text()
        if is_verified:
            badge_text.append("\n  ✔ ON-CHAIN VERIFICATION CONFIRMED  \n\n", style="bold bright_white on dark_green")
            badge_text.append(f"Content Hash: {content_hash}\n", style="bold cyan")
            badge_text.append(f"Smart contract state matches local canonical digest with 100% cryptographic integrity.\n", style="white")
            self.console.print(Panel(badge_text, border_style="green", box=box.HEAVY, expand=True))
        else:
            badge_text.append("\n  ✖ ON-CHAIN VERIFICATION FAILED  \n\n", style="bold bright_white on dark_red")
            badge_text.append(f"Content Hash '{content_hash}' could not be verified on-chain.\n", style="bold yellow")
            self.console.print(Panel(badge_text, border_style="red", box=box.HEAVY, expand=True))

    def render_tamper_matrix(self, tamper_report: Dict[str, Any]) -> None:
        """Renders 5-scenario tamper attack simulation table."""
        table = Table(
            title="Automated 5-Scenario Tamper Detection Matrix",
            box=box.ROUNDED,
            expand=True,
            show_lines=True,
        )
        table.add_column("#", style="bold white", width=4)
        table.add_column("Tamper Scenario", style="bold yellow", width=28)
        table.add_column("Altered Field", style="cyan", width=18)
        table.add_column("Field Diff Description", style="white")
        table.add_column("Detection Status", style="bold", width=18)

        scenarios = tamper_report.get("scenarios", [])
        for i, s in enumerate(scenarios, 1):
            status_style = "[bold green]TAMPER DETECTED[/bold green]" if s.get("status") == "TAMPER_DETECTED" else "[bold red]FAILED[/bold red]"
            diffs = s.get("diffs", [])
            field_name = diffs[0].get("field_name", "N/A") if diffs else "N/A"
            diff_desc = diffs[0].get("impact_description", "Value modified") if diffs else "Hash mismatch"

            table.add_row(
                str(i),
                s.get("scenario_name", f"Scenario {i}"),
                field_name,
                diff_desc,
                status_style,
            )

        self.console.print(table)

    def render_execution_summary(self, summary: Dict[str, Any]) -> None:
        """Renders final execution summary table."""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Stage #", style="bold white", width=8)
        table.add_column("Stage Name", style="bold yellow", width=34)
        table.add_column("Duration", style="cyan", width=12)
        table.add_column("Status", style="bold", width=12)

        stages = summary.get("stages", [])
        for st in stages:
            num = f"[{st.get('stage_number', 0):02d}]"
            name = st.get("stage_name", "Stage")
            dur = f"{st.get('duration_seconds', 0.0):.3f}s"
            status = st.get("status", "SUCCESS")
            status_style = "[green]SUCCESS[/green]" if status == "SUCCESS" else "[yellow]ABORTED[/yellow]" if status == "ABORTED" else "[red]FAILED[/red]"
            table.add_row(num, name, dur, status_style)

        total_dur = summary.get("total_duration_seconds", 0.0)
        overall_status = summary.get("status", "SUCCESS")

        footer = Text()
        footer.append(f"Total Pipeline Duration: {total_dur:.3f}s | Overall Status: ", style="bold white")
        footer.append(f"{overall_status}\n", style="bold green" if overall_status == "SUCCESS" else "bold yellow")
        footer.append(f"Artifacts persisted to: {summary.get('pipeline_log_path', 'artifacts/')}", style="dim white")

        self.console.print(Panel(table, title="[bold cyan]🏁 End-to-End Execution Summary[/bold cyan]", subtitle=footer, border_style="cyan"))
