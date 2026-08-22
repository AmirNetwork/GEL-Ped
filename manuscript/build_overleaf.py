# Author: Amir Ghorbani
"""Build the integrated GEL-Ped Elsevier/Overleaf manuscript.

Author: Amir Ghorbani
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript" / "manuscript.md"
FIGURES = ROOT / "figures"
PROJECT = ROOT / "manuscript" / "overleaf_trc"

FIGURE_INSERTS = {
    "Figure 1 grounds the evaluation": (
        "experimental_context_and_protocol.png",
        "Real experiments and frozen evaluation protocol. Overhead views show the corridor "
        "and crossing facilities. Seven complete runs calibrate the models; five familiar "
        "corridor runs, three altered-geometry runs, and thirteen perpendicular-crossing "
        "runs remain untouched. Images: Juelich Pedestrian Dynamics Data Archive, CC BY 4.0; "
        "DOIs 10.34735/ped.2013.5 and 10.34735/ped.2013.4.",
    ),
    "Figure 2 summarizes GEL-Ped": (
        "tensor_residual_architecture.png",
        "GEL-Ped dual-expert architecture. A direct neural expert learns the full "
        "mapping in the goal-aligned frame. A structured expert starts from the tensor "
        "response and learns a support-gated nonlinear residual. A run-wise calibrated "
        "convex blend combines their complementary predictions.",
    ),
    "Figure 3 reports the primary comparison": (
        "primary_performance_summary.png",
        "Primary accuracy results. Bars show run-balanced velocity RMSE and standard errors "
        "for the three untouched regimes. Run-level dots and bootstrap intervals show the "
        "GEL-Ped reductions relative to constant velocity, calibrated Social Force, and "
        "the unrestricted linear model.",
    ),
    "Figure 4 tests whether the crossing advantage": (
        "cross_horizon_neural_transfer.png",
        "Cross-horizon topology transfer. Left: run-balanced RMSE on thirteen unseen "
        "perpendicular-crossing runs for the direct network, a protocol-matched 2024 "
        "goal-stable hybrid, and GEL-Ped. Right: complete-run "
        "reductions relative to the direct network with run-bootstrap intervals. Model "
        "settings are frozen rather than retuned by horizon.",
    ),
    "Figure 5 isolates estimation efficiency": (
        "data_efficiency.png",
        "Fixed-design estimation efficiency. Left: pooled held-out RMSE as the number of "
        "site-specific fitting runs increases; shading spans all subsets. Right: reduction "
        "from the matched direct neural expert for every subset (circles) and the subset "
        "mean (diamonds). All 127 subsets of seven runs are enumerated.",
    ),
    "Figure 6 examines model components": (
        "hybrid_diagnostics.png",
        "Component, support, coefficient, and seed diagnostics. The direct network dominates "
        "matched-corridor interpolation, while the structured and blended predictors retain "
        "more of their accuracy under crossing transfer. The support gate is observable and "
        "optimization-seed ranges do not reverse the crossing pattern.",
    ),
    "Figure 7 shows a representative crossing frame": (
        "external_crossing_hybrid_snapshot.png",
        "Illustrative unseen-crossing frame. Arrows compare observed future velocity with "
        "constant-velocity, direct-network, and GEL-Ped forecasts. The frame is "
        "selected reproducibly as the median positive GEL-Ped gain among frames containing "
        "at least twenty pedestrians; aggregate run-level tests provide the inferential evidence.",
    ),
}

CITATIONS = {
    "Alahi et al., 2016": "alahi2016",
    "Boltes et al., 2020": "boltes2020",
    "Boltes and Seyfried, 2013": "boltes2013",
    "Cao et al., 2017": "cao2017",
    "Chraibi et al., 2011": "chraibi2011",
    "Dietrich and Koster, 2014": "dietrich2014",
    "Feliciani et al., 2023": "feliciani2023",
    "Ghorbani, 2022": "ghorbani2022",
    "Gupta et al., 2018": "gupta2018",
    "Haghani, 2020": "haghani2020",
    "Haghani, 2023": "haghani2023",
    "Haghani and Sarvi, 2018": "haghani2018",
    "Hartmann, 2010": "hartmann2010",
    "Helbing and Molnar, 1995": "helbing1995",
    "Hughes, 2002": "hughes2002",
    "Johansson et al., 2007": "johansson2007",
    "Karamouzas et al., 2014": "karamouzas2014",
    "Korbmacher et al., 2024": "korbmacher2024",
    "Liu et al., 2024": "liu2024",
    "Mai et al., 2025": "mai2025",
    "Mohamed et al., 2020": "mohamed2020",
    "Moussaid et al., 2011": "moussaid2011",
    "Munir and Kucner, 2025": "munir2025",
    "Pouw et al., 2024": "pouw2024",
    "Salzmann et al., 2020": "salzmann2020",
    "Sang et al., 2024": "sang2024",
    "Su et al., 2024": "su2024",
    "Wang et al., 2025": "wang2025",
    "Wang et al., 2024": "wang2024dynamics",
    "van den Berg et al., 2008": "vandenberg2008",
    "Xu et al., 2021": "xu2021",
    "Xu et al., 2024": "xu2024",
    "Xu et al., 2026": "xu2026",
    "Zanlungo et al., 2011": "zanlungo2011",
}


def normalize_ascii_hyphens(text: str) -> str:
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )


def citation_placeholders(text: str, placeholders: dict[str, str]) -> str:
    def reserve(value: str) -> str:
        token = f"@@PH{len(placeholders)}@@"
        placeholders[token] = value
        return token

    text = text.replace(
        "Haghani and Sarvi's (2018)", reserve(r"\citet{haghani2018}'s")
    )

    textual = sorted(CITATIONS.items(), key=lambda item: len(item[0]), reverse=True)
    for label, key in textual:
        author, year = label.rsplit(", ", 1)
        text = text.replace(f"{author} ({year})", reserve(rf"\citet{{{key}}}"))

    pattern = re.compile(r"\(([^()]*(?:19|20)\d{2}[^()]*)\)")

    def parenthetical(match: re.Match[str]) -> str:
        labels = [part.strip() for part in match.group(1).split(";")]
        if labels and all(label in CITATIONS for label in labels):
            keys = ",".join(CITATIONS[label] for label in labels)
            return reserve(rf"\citep{{{keys}}}")
        return match.group(0)

    return pattern.sub(parenthetical, text)


def inline_latex(text: str) -> str:
    placeholders: dict[str, str] = {}

    def reserve(value: str) -> str:
        token = f"@@PH{len(placeholders)}@@"
        placeholders[token] = value
        return token

    text = normalize_ascii_hyphens(text)
    text = citation_placeholders(text, placeholders)
    text = re.sub(
        r"\bFigure\s+([0-9]+)",
        lambda m: reserve(
            rf"\hyperref[fig:{m.group(1)}]{{Figure~\ref*{{fig:{m.group(1)}}}}}"
        ),
        text,
    )
    text = re.sub(
        r"\bTable\s+((?:[A-Z]\.)?[0-9]+)",
        lambda m: reserve(
            rf"\hyperref[tab:{m.group(1).replace('.', '')}]"
            rf"{{Table~{m.group(1)}}}"
        ),
        text,
    )
    text = re.sub(
        r"\bAppendix\s+([A-Z])",
        lambda m: reserve(
            rf"\hyperref[app:{m.group(1)}]{{Appendix~{m.group(1)}}}"
        ),
        text,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: reserve("$" + m.group(1) + "$"),
        text,
    )
    text = text.replace("⁻²", reserve(r"\textsuperscript{-2}"))
    text = text.replace("⁻¹⁷", reserve(r"\textsuperscript{-17}"))
    text = text.replace("⁻¹⁵", reserve(r"\textsuperscript{-15}"))
    text = text.replace("²", reserve(r"\textsuperscript{2}"))
    text = text.replace("10^{-3}", reserve(r"$10^{-3}$"))
    text = re.sub(
        r"https?://[^\s]+",
        lambda m: reserve(r"\url{" + m.group(0).rstrip(".") + "}")
        + ("." if m.group(0).endswith(".") else ""),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda m: reserve(r"\path{" + m.group(1) + "}"),
        text,
    )
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: reserve(r"\textbf{" + inline_latex(m.group(1)) + "}"),
        text,
    )
    text = re.sub(
        r"\*([^*]+?)\*",
        lambda m: reserve(r"\emph{" + inline_latex(m.group(1)) + "}"),
        text,
    )
    text = escape_plain(text)
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def escape_plain(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def strip_number(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()


def latex_table(
    rows: list[list[str]], caption: str, number: str, *, floating: bool = True
) -> str:
    columns = len(rows[0])
    landscape = columns >= 8
    environment = "sidewaystable" if landscape else "table"
    width = r"\textheight" if landscape else r"\textwidth"
    align = "l" + "c" * (columns - 1)
    body = []
    if floating:
        body.append(rf"\begin{{{environment}}}[p]" if landscape else r"\begin{table}[!htbp]")
        body.append(r"\centering")
        body.append(r"\caption{" + inline_latex(caption) + "}")
        body.append(rf"\label{{tab:{number.replace('.', '')}}}")
    else:
        body.append(r"\noindent\begin{minipage}{\textwidth}")
        body.append(r"\centering")
        body.append(rf"\phantomsection\label{{tab:{number.replace('.', '')}}}")
        body.append(r"\textbf{Table " + str(number) + ". " + inline_latex(caption) + r"}\\[3pt]")
    if columns <= 4 and not landscape:
        body.append(r"\small")
        body.append(r"\renewcommand{\arraystretch}{1.13}")
        body.append(r"\setlength{\tabcolsep}{5pt}")
        column_spec = (
            r">{\raggedright\arraybackslash}p{0.20\textwidth}"
            + rf"*{{{columns - 1}}}{{>{{\raggedright\arraybackslash}}X}}"
        )
        body.append(rf"\begin{{tabularx}}{{\textwidth}}{{{column_spec}}}")
    else:
        body.append(r"\scriptsize" if landscape else r"\footnotesize")
        body.append(r"\renewcommand{\arraystretch}{1.10}")
        body.append(r"\setlength{\tabcolsep}{4pt}")
        body.append(rf"\resizebox{{{width}}}{{!}}{{%")
        body.append(rf"\begin{{tabular}}{{{align}}}")
    body.append(r"\toprule")
    body.append(" & ".join(inline_latex(cell) for cell in rows[0]) + r" \\")
    body.append(r"\midrule")
    for row in rows[1:]:
        body.append(" & ".join(inline_latex(cell) for cell in row) + r" \\")
    body.append(r"\bottomrule")
    if columns <= 4 and not landscape:
        body.append(r"\end{tabularx}")
    else:
        body.append(r"\end{tabular}")
        body.append(r"}")
    if floating:
        body.append(rf"\end{{{environment}}}")
    else:
        body.append(r"\end{minipage}")
    return "\n".join(body)


def latex_figure(filename: str, caption: str, number: int) -> str:
    width = "0.98" if number == 1 else "0.95"
    return "\n".join(
        [
            r"\begin{figure}[!htbp]",
            r"\centering",
            rf"\includegraphics[width={width}\textwidth]{{figures/{filename}}}",
            r"\caption{" + inline_latex(caption) + "}",
            rf"\label{{fig:{number}}}",
            r"\end{figure}",
        ]
    )


def extract_frontmatter(lines: list[str]) -> tuple[str, str, list[str], int]:
    title = lines[0][2:].strip()
    abstract_index = lines.index("## Abstract")
    intro_index = lines.index("## 1. Introduction")
    abstract_parts = []
    keywords = []
    for line in lines[abstract_index + 1 : intro_index]:
        if line.startswith("**Keywords:**"):
            keywords = [item.strip(" *") for item in line.split(":", 1)[1].split(";")]
        elif line.strip():
            abstract_parts.append(line.strip())
    return title, " ".join(abstract_parts), keywords, intro_index


def build_body(lines: list[str], start: int) -> str:
    output: list[str] = []
    index = start
    table_caption: tuple[str, str] | None = None
    figure_number = 0
    equation_number = 0
    appendix_started = False

    while index < len(lines):
        line = lines[index]
        if line == "## References":
            break
        if not line.strip():
            index += 1
            continue
        if line == r"\[":
            equation_number += 1
            equation = []
            index += 1
            while index < len(lines) and lines[index] != r"\]":
                if lines[index].strip():
                    equation.append(normalize_ascii_hyphens(lines[index]).strip())
                index += 1
            if equation_number == 1:
                split_lines = [equation[0] + r"\\"]
                split_lines.extend("&" + item + (r"\\" if offset < len(equation) - 1 else "") for offset, item in enumerate(equation[1:], start=1))
                equation_block = [
                    r"\begin{equation}",
                    rf"\label{{eq:{equation_number}}}",
                    r"\begin{split}",
                    *split_lines,
                    r"\end{split}",
                    r"\end{equation}",
                ]
            else:
                equation_block = [
                    r"\begin{equation}",
                    rf"\label{{eq:{equation_number}}}",
                    *equation,
                    r"\end{equation}",
                ]
            output.append("\n".join(equation_block))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                values = [value.strip() for value in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", value) for value in values):
                    rows.append(values)
                index += 1
            if table_caption is None:
                raise ValueError(f"Table at line {index} has no caption")
            output.append(
                latex_table(
                    rows,
                    table_caption[1],
                    table_caption[0],
                    floating=not appendix_started,
                )
            )
            table_caption = None
            continue
        heading = re.match(r"^(#{2,4})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            title = strip_number(heading.group(2))
            appendix = re.match(r"^Appendix\s+([A-Z])\.\s+(.*)$", heading.group(2))
            if level == 2 and appendix:
                if not appendix_started:
                    output.extend(
                        [
                            r"\clearpage",
                            r"\appendix",
                            r"\counterwithin{table}{section}",
                        ]
                    )
                    appendix_started = True
                output.append(
                    r"\section{" + inline_latex(appendix.group(2)) + "}"
                    + rf"\label{{app:{appendix.group(1)}}}"
                )
                index += 1
                continue
            starred = title in {
                "Data availability",
                "Code availability",
                "Ethics statement",
                "Declaration of competing interest",
                "Acknowledgements",
            }
            command = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            output.append(rf"\{command}{'*' if starred else ''}{{{inline_latex(title)}}}")
            index += 1
            continue

        paragraph_lines = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not (
            lines[index].startswith("#")
            or lines[index].startswith("|")
            or lines[index] == r"\["
        ):
            paragraph_lines.append(lines[index].strip())
            index += 1
        paragraph = " ".join(paragraph_lines)
        caption_match = re.match(
            r"^Table\s+((?:[A-Z]\.)?\d+)\.\s+(.*)$", paragraph
        )
        if caption_match:
            table_caption = (caption_match.group(1), caption_match.group(2))
            continue
        output.append(inline_latex(paragraph))
        for prefix, (filename, caption) in FIGURE_INSERTS.items():
            if paragraph.startswith(prefix):
                figure_number += 1
                output.append(latex_figure(filename, caption, figure_number))
                break

    if figure_number != 7:
        raise ValueError(f"Expected 7 figures, inserted {figure_number}")
    return "\n\n".join(output)


def build_main() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title, abstract, keywords, start = extract_frontmatter(lines)
    body = build_body(lines, start)
    header = r"""\documentclass[preprint,11pt,authoryear]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array}
\usepackage{rotating}
\usepackage{microtype}
\usepackage{lineno}
\usepackage[a4paper,left=1.55cm,right=1.55cm,top=1.70cm,bottom=1.70cm]{geometry}
\usepackage[dvipsnames]{xcolor}
\usepackage[colorlinks=true,linkcolor=MidnightBlue,citecolor=MidnightBlue,urlcolor=MidnightBlue]{hyperref}

\modulolinenumbers[5]
\graphicspath{{figures/}}
\setlength{\textfloatsep}{11pt plus 2pt minus 2pt}
\setlength{\floatsep}{9pt plus 2pt minus 2pt}
\setcounter{topnumber}{3}
\renewcommand{\topfraction}{0.92}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.82}

\begin{document}

\begin{frontmatter}
"""
    frontmatter = "\n".join(
        [
            r"\title{" + inline_latex(title) + "}",
            r"\author[melb]{Amir Ghorbani\corref{cor1}}",
            r"\cortext[cor1]{Corresponding author}",
            r"\ead{ghorbania@student.unimelb.edu.au}",
            r"\affiliation[melb]{organization={Transport Engineering Group, Department of Infrastructure Engineering, The University of Melbourne},addressline={Grattan Street},city={Parkville},state={VIC},postcode={3010},country={Australia}}",
            r"\begin{abstract}",
            inline_latex(abstract),
            r"\end{abstract}",
            r"\begin{keyword}",
            " \\sep ".join(inline_latex(keyword) for keyword in keywords),
            r"\end{keyword}",
            r"\end{frontmatter}",
            r"\linenumbers",
        ]
    )
    footer = r"""

\bibliographystyle{elsarticle-harv}
\bibliography{references}

\end{document}
"""
    return header + frontmatter + "\n\n" + body + footer


def main() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    project_figures = PROJECT / "figures"
    project_figures.mkdir(exist_ok=True)
    for filename, _ in FIGURE_INSERTS.values():
        shutil.copy2(FIGURES / filename, project_figures / filename)
    shutil.copy2(
        FIGURES / "tensor_residual_architecture.png",
        project_figures / "graphical_abstract.png",
    )
    shutil.copy2(SOURCE, PROJECT / "source_manuscript.md")
    (PROJECT / "main.tex").write_text(build_main(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
