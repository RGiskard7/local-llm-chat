#!/usr/bin/env python3
"""
Local LLM Chat - Punto de Entrada
Lanzador simple que delega al módulo CLI
"""

from src.local_llm_chat.cli import run_cli

if __name__ == "__main__":
    run_cli()
