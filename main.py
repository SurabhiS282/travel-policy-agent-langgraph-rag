import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from src.config import (
    MAX_RETRIES,
    LLM_PROVIDER,
    GROQ_API_KEY,
    OLLAMA_MODEL_NAME,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
)
from src.graph import create_travel_policy_graph
from src.ingestion import build_vectorstore


console = Console(highlight=False)


def print_banner(debug_mode: bool):
    """Prints the application startup banner."""
    provider_str = f"[cyan]{LLM_PROVIDER.upper()}[/cyan]"
    debug_str = "[green]ON[/green]" if debug_mode else "[dim]OFF[/dim]"
    
    banner_text = (
        f"[bold white]AIRLINE TRAVEL POLICY AGENT[/bold white]\n"
        f"[dim]Powered by Python, LangGraph, RAG, and LangSmith[/dim]\n\n"
        f"• LLM Provider: {provider_str}\n"
        f"• Node Tracing: {debug_str} (type [yellow]:debug[/yellow] to toggle)\n"
        f"• Type [yellow]'exit'[/yellow] or [yellow]'quit'[/yellow] to end the session."
    )
    console.print(Panel(banner_text, border_style="cyan", expand=False))


def check_api_keys():
    """Validates that necessary configuration is present."""
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        console.print(
            Panel(
                "[bold yellow][!] Free Groq API Key is missing in .env[/bold yellow]\n\n"
                "1. Get a 100% free API key at: [cyan]https://console.groq.com/keys[/cyan]\n"
                "2. Add to your [yellow].env[/yellow] file:\n"
                "   [cyan]GROQ_API_KEY=\"gsk_...\"[/cyan]\n\n"
                "💡 Alternatively, set [cyan]LLM_PROVIDER=\"ollama\"[/cyan] in [yellow].env[/yellow] to run completely locally without any API key!",
                title="Free Model Setup",
                border_style="yellow",
            )
        )
    elif LLM_PROVIDER == "ollama":
        console.print(
            Panel(
                f"[bold green][v] Using Local Offline Ollama model: '{OLLAMA_MODEL_NAME}'[/bold green]\n"
                f"Ensure Ollama is running (`ollama run {OLLAMA_MODEL_NAME}`).",
                title="Local Offline Mode",
                border_style="green",
            )
        )
    elif LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        console.print(
            Panel(
                "[bold yellow][!] GOOGLE_API_KEY is not configured![/bold yellow]\n\n"
                "Get a free Gemini API key at: [cyan]https://aistudio.google.com/app/apikey[/cyan]\n"
                "And add to [yellow].env[/yellow]:\n"
                "[cyan]GOOGLE_API_KEY=\"AIza...\"[/cyan]",
                title="Free Gemini Setup",
                border_style="yellow",
            )
        )



def main():
    console.clear()
    debug_mode = True  # Default ON to showcase LangGraph node progression

    print_banner(debug_mode)
    check_api_keys()

    console.print("\n[dim]Initializing Vector Store and LangGraph workflow...[/dim]")
    try:
        build_vectorstore(force_rebuild=False)
        app = create_travel_policy_graph()
        console.print("[bold green]✓ LangGraph workflow successfully compiled and ready![/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        return

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold green]👋 Thank you for using Travel Policy Agent. Safe travels![/bold green]")
                break

            if user_input.lower() == ":debug":
                debug_mode = not debug_mode
                status = "[green]ENABLED[/green]" if debug_mode else "[yellow]DISABLED[/yellow]"
                console.print(f"[dim]Node tracing {status}.[/dim]")
                continue

            # Initial state setup
            initial_state = {
                "original_question": user_input,
                "rewritten_question": "",
                "retrieved_docs": [],
                "context_text": "",
                "is_context_relevant": False,
                "answer": "",
                "is_answer_grounded": False,
                "evaluation_feedback": "",
                "retry_count": 0,
                "max_retries": MAX_RETRIES,
                "trace_logs": [],
            }

            console.print("\n[yellow]⏳ Executing LangGraph workflow...[/yellow]")
            
            # Execute the graph
            final_state = app.invoke(initial_state)

            # Display Debug Trace if enabled
            if debug_mode and final_state.get("trace_logs"):
                table = Table(title="🔍 LangGraph Step-by-Step Execution Trace", border_style="dim")
                table.add_column("Step", justify="right", style="cyan", no_wrap=True)
                table.add_column("Node Event / State Transition", style="white")

                for i, log in enumerate(final_state["trace_logs"], 1):
                    table.add_row(str(i), log)
                console.print(table)

            # Display Final Answer
            answer = final_state.get("answer", "No response generated.")
            console.print(Panel(Markdown(answer), title="[bold green]✈️  Travel Policy Agent[/bold green]", border_style="green"))

        except KeyboardInterrupt:
            console.print("\n\n[bold green]👋 Session ended. Goodbye![/bold green]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error during graph execution:[/bold red] {e}")


if __name__ == "__main__":
    main()
