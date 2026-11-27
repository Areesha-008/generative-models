"""F.1 One web app, four tabs, all running the trained models.

From the repo root:

    python app/app.py            # http://127.0.0.1:7860
    python app/app.py --share    # also prints a public link (use this on Colab)

Each model loads the first time its tab is used.
"""
import argparse
import html
import random

import gradio as gr

import backend as be

MAE_TILES = 5400  # EuroSAT test split size; the index wraps if it is smaller


def friendly(fn):
    """Show a missing checkpoint, dataset or index as a message in the page."""
    def wrapped(*args):
        try:
            return fn(*args)
        except be.MissingFile as err:
            raise gr.Error(str(err), duration=None) from err
    return wrapped


def character_names():
    try:
        return be.uhat_names()
    except be.MissingFile:
        return []


# ------------------------------------------------------------------------ RAG

def render_passages(passages):
    blocks = []
    for p in sorted(passages, key=lambda p: (not p["cited"], p["n"])):
        tag = "cited" if p["cited"] else "retrieved, not cited"
        blocks.append(
            "<div class='passage'>"
            f"<div class='passage-head'><b>[{p['n']}]</b> abstract id "
            f"<code>{p['abstract_id']}</code> · score {p['score']:.3f} · {tag}</div>"
            f"<div>{html.escape(p['text'])}</div>"
            "</div>"
        )
    return "".join(blocks)


def rag_tab(question):
    if not question.strip():
        raise gr.Error("Type a question first.")
    result = be.rag_answer(question)
    answer = html.escape(result["answer"]).replace("\n", "<br>")
    answer_html = (
        f"<div class='answer'>{answer}</div>"
        f"<div class='notice'>⚠️ {html.escape(result['notice'])}</div>"
    )
    cited = sum(p["cited"] for p in result["passages"])
    heading = f"**Passages** — {cited} of {len(result['passages'])} cited"
    if result["idk"]:
        heading += " (the model said it did not know)"
    return answer_html, heading, render_passages(result["passages"])


CSS = """
.answer {font-size: 1.05rem; line-height: 1.5;}
.notice {margin-top: .75rem; padding: .5rem .75rem; border-left: 4px solid #d97706;
         font-weight: 600;}
.passage {margin: .5rem 0 1rem; line-height: 1.45;}
.passage-head {margin-bottom: .25rem; opacity: .85;}
"""


def build():
    names = character_names()
    first, second = (names[0], names[1]) if len(names) > 1 else (None, None)

    with gr.Blocks(title="Four generative models") as demo:
        gr.Markdown(
            "# Four generative models\n"
            f"MAE on EuroSAT · VAE and DDPM on UHaT Urdu characters · RAG on BioASQ. "
            f"Running on `{be.DEVICE}`."
        )

        with gr.Tab("MAE"):
            gr.Markdown(
                "Pick a EuroSAT test tile. The encoder sees 16 of its 64 patches "
                "(grey = hidden); the decoder fills in the rest. The linear probe on the "
                "frozen encoder predicts the land-cover class from the whole tile."
            )
            with gr.Row():
                tile = gr.Slider(0, MAE_TILES - 1, value=0, step=1, label="Test tile")
                mask_seed = gr.Number(value=0, precision=0, label="Mask seed")
                with gr.Column(scale=0, min_width=140):
                    random_tile = gr.Button("Random tile")
                    run_mae = gr.Button("Reconstruct", variant="primary")
            with gr.Row():
                masked = gr.Image(label="Masked input (75% hidden)", type="pil")
                recon = gr.Image(label="Reconstruction", type="pil")
                original = gr.Image(label="Original", type="pil")
            with gr.Row():
                predicted = gr.Label(label="Predicted land-cover class", num_top_classes=3)
                true_class = gr.Textbox(label="True class", interactive=False)
            mae_out = [masked, recon, original, predicted, true_class]
            run_mae.click(friendly(be.mae_demo), [tile, mask_seed], mae_out)
            random_tile.click(
                lambda: random.randrange(MAE_TILES), None, tile
            ).then(friendly(be.mae_demo), [tile, mask_seed], mae_out)

        with gr.Tab("VAE") as vae_tab:
            gr.Markdown("New characters decoded from random codes z ~ N(0, I).")
            new_chars = gr.Button("New random characters", variant="primary")
            random_grid = gr.Image(label="16 samples", type="pil")
            new_chars.click(
                friendly(lambda: be.vae_random(random.randrange(2**31))), None, random_grid
            )

            gr.Markdown(
                "Interpolate between two characters: both are encoded to their mean "
                "code mu, and the decoder draws (1 − w)·mu_A + w·mu_B."
            )
            with gr.Row():
                char_a = gr.Dropdown(names, value=first, label="Character A")
                char_b = gr.Dropdown(names, value=second, label="Character B")
                example = gr.Number(value=0, precision=0, label="Which test image of each")
            weight = gr.Slider(0, 1, value=0.5, step=0.05, label="w (0 = A, 1 = B)")
            with gr.Row():
                img_a = gr.Image(label="A", type="pil")
                blend = gr.Image(label="Decoded blend", type="pil")
                img_b = gr.Image(label="B", type="pil")
            strip = gr.Image(label="A → B in 10 steps", type="pil")
            interp_in = [char_a, char_b, weight, example]
            interp_out = [img_a, blend, img_b, strip]
            for control in (char_a, char_b, example):
                control.change(friendly(be.vae_interpolate), interp_in, interp_out)
            weight.release(friendly(be.vae_interpolate), interp_in, interp_out)
            vae_tab.select(friendly(be.vae_interpolate), interp_in, interp_out)

        with gr.Tab("Diffusion"):
            gr.Markdown(
                "Pick a character; the class-conditional DDPM draws 8 new samples "
                "with 1000 denoising steps. This takes a few seconds on a GPU and "
                "longer on a CPU."
            )
            with gr.Row():
                char = gr.Dropdown(names, value=first, label="Character")
                seed = gr.Number(value=0, precision=0, label="Seed")
                generate = gr.Button("Generate 8 samples", variant="primary")
            samples = gr.Image(label="8 samples", type="pil")
            generate.click(friendly(be.diffusion_generate), [char, seed], samples)

        with gr.Tab("RAG"):
            gr.Markdown(
                "Ask a biomedical research question. MiniLM retrieves the top 5 chunks "
                "of BioASQ abstracts, and Qwen2.5-1.5B-Instruct answers from them only, "
                "citing passage numbers."
            )
            question = gr.Textbox(label="Question", lines=2,
                                  placeholder="e.g. What is the mechanism of action of ibalizumab?")
            ask = gr.Button("Answer", variant="primary")
            gr.Examples(
                [
                    ["What is the mechanism of action of ibalizumab?"],
                    ["Is tubulin acetylation involved in cell motility?"],
                    ["What is Neu5Gc?"],
                ],
                inputs=question,
            )
            answer = gr.HTML(label="Answer")
            passages_heading = gr.Markdown()
            passages = gr.HTML()
            rag_out = [answer, passages_heading, passages]
            ask.click(friendly(rag_tab), question, rag_out)
            question.submit(friendly(rag_tab), question, rag_out)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="also print a public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build().queue().launch(share=args.share, server_port=args.port, css=CSS)


if __name__ == "__main__":
    main()
