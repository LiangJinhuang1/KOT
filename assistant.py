#!/usr/bin/env python3
"""Interactive thesis assistant — executor: Sonnet 4.6, advisor: Opus 4.7."""

import os
import sys
import anthropic

SYSTEM_PROMPT = """You are a research assistant for a bioinformatics PhD thesis on single-cell multiomics data integration.

Project datasets:
- BMMC CITE-seq (GSE194122): bone marrow mononuclear cells, RNA (GEX) + protein (ADT)
- BMMC Multiome (GSE194122): bone marrow, RNA (GEX) + ATAC chromatin accessibility
- PBMC CITE-seq (10x Genomics 5k PBMC): peripheral blood, RNA + antibody capture
- Perturb-CITE-seq (SCP1064): perturbation experiment, RNA + protein

Codebase structure:
- scr/Datasets/Preprocessing.py — loads and splits AnnData into RNA/protein/ATAC modalities,
  with disk caching via load_and_preprocess_cached() and load_perturb_cite_seq_files_cached()
- train.py — defines TEST_INPUTS and run_one() / run_many() for testing preprocessing
- main.py — entry point calling run_many(TEST_INPUTS)

Stack: scanpy (sc), anndata (ad), pandas. Data stored as AnnData (.h5ad), accessed via .X, .obs, .var, .obsm, .layers.

Your role:
1. Debug and write Python code for this project — be specific about variable names, function signatures, and file locations
2. Help with bioinformatics analysis: normalization (log1p, CLR, TF-IDF), dimensionality reduction (PCA, LSI), clustering (Leiden), UMAP, batch correction, multimodal integration
3. Explain single-cell genomics and statistics concepts clearly and intuitively when confused
4. Suggest and evaluate analytical approaches for multimodal single-cell data

Always be concrete — reference actual project files and data structures when relevant."""

ADVISOR_TOOL = {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-7"}


def run():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic()
    messages: list = []
    show_advisor = "--verbose" in sys.argv

    print("Thesis Assistant  (Sonnet 4.6 + Opus 4.7 advisor)")
    print("Commands: 'clear' to reset, 'quit' to exit", end="")
    if not show_advisor:
        print("  |  run with --verbose to see advisor guidance", end="")
    print("\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages.clear()
            print("[Conversation cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            print("\nAssistant: ", end="", flush=True)
            advisor_active = False

            with client.beta.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                betas=["advisor-tool-2026-03-01"],
                tools=[ADVISOR_TOOL],
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                current_block_type = None

                for event in stream:
                    if event.type == "content_block_start":
                        block_type = getattr(event.content_block, "type", None)
                        current_block_type = block_type

                        if block_type == "server_tool_use" and not advisor_active:
                            print("\n\033[2m[consulting Opus advisor...]\033[0m ", end="", flush=True)
                            advisor_active = True

                        elif block_type == "advisor_tool_result" and show_advisor:
                            print("\n\033[2m[Advisor guidance]\033[0m\n", flush=True)

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        delta_type = getattr(delta, "type", None)

                        if delta_type == "text_delta":
                            if current_block_type == "text":
                                print(delta.text, end="", flush=True)
                            elif current_block_type == "advisor_tool_result" and show_advisor:
                                # advisor_tool_result deltas carry the advisor's guidance text
                                advisor_text = getattr(delta, "text", None)
                                if advisor_text:
                                    print(f"\033[2m{advisor_text}\033[0m", end="", flush=True)

                    elif event.type == "content_block_stop":
                        if current_block_type == "advisor_tool_result" and show_advisor:
                            print("\n\033[2m[end advisor guidance]\033[0m\n", flush=True)
                        current_block_type = None

                final = stream.get_final_message()

            print("\n")
            # Preserve full content list (including advisor blocks) for multi-turn context
            messages.append({"role": "assistant", "content": final.content})

        except anthropic.AuthenticationError:
            print("\nError: Invalid API key.")
            messages.pop()
        except anthropic.BadRequestError as e:
            print(f"\nBad request: {e.message}")
            messages.pop()
        except anthropic.APIError as e:
            print(f"\nAPI error: {e}")
            messages.pop()


if __name__ == "__main__":
    run()
