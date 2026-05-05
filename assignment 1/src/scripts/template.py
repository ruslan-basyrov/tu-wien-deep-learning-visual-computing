from jinja2 import Environment, FileSystemLoader
import argparse
from pathlib import Path

TEMPLATE_FILENAME = "plot_figure.js.jinja"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Construct proper JS scripts for Observable Plot")
    parser.add_argument("input", help="Directory with templates of JS files")
    parser.add_argument("output", help="Directory to store built JS files and SVG")
    parser.add_argument("fig", help="Base file name of the plot")
    parser.add_argument("fig_dir", help="Directory to store rendered SVG of figure")

    args = parser.parse_args()

    fig_path = Path(args.fig_dir, f"{args.fig}.svg")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader([args.output, args.input]))

    template = env.get_template(TEMPLATE_FILENAME)

    for format in ("pdf", "html"):
        result = template.render(fig_path=fig_path, fig=args.fig, format=format)
        filename = f"{args.fig}.js"
        if format == "pdf":
            filename = "svg_" + filename
        output_path = Path(args.output, filename)
        output_path.write_text(result, encoding="utf-8")

    chunk_template = env.get_template("chunk.qmd.jinja")
    chunk_result = chunk_template.render(fig=args.fig)
    Path(args.output, f"{args.fig}.qmd").write_text(chunk_result, encoding="utf-8")
