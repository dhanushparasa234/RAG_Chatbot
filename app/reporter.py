import os
import json
import unicodedata
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import get_llm
from fpdf import FPDF

class ExecutivePDF(FPDF):
    def header(self):
        # Professional dark slate header banner on every page
        self.set_fill_color(30, 41, 59) # Slate-800
        self.rect(0, 0, 210, 22, 'F')
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 6)
        self.cell(0, 10, 'EXECUTIVE FINANCIAL INTELLIGENCE & AUDIT REPORT', 0, 1, 'L')
        self.set_y(28)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def sanitize_text(text: str) -> str:
    """
    Normalizes unicode characters, cleans HTML entities, replaces math symbols,
    and strictly encodes to ASCII to prevent any rendering or encoding bugs.
    """
    if not isinstance(text, str):
        text = str(text)

    text = (text.replace('&#8208;', '-')
                .replace('&ndash;', '-')
                .replace('&mdash;', '-')
                .replace('×', ' times ')
                .replace('÷', ' divided by '))

    text = unicodedata.normalize("NFKD", text)

    replacements = {
        '₹': 'INR ',
        '–': '-',
        '—': '-',
        '•': '-',
        '·': '-',
        '≈': 'approximately ',
        '±': '+/-',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '…': '...',
        '\xa0': ' ',
        '\u200b': ''
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
        
    return text.encode('ascii', 'ignore').decode('ascii')


def generate_executive_report(run_dir: str) -> dict:
    """
    Generates professional Markdown and compiles an exhaustive, token-safe executive PDF report.
    """
    facts_path = os.path.join(run_dir, "financial_facts.json")
    if not os.path.exists(facts_path):
        raise FileNotFoundError(f"Financial facts JSON not found at {facts_path}")

    with open(facts_path, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # 1. Invoke LLM with a token-optimized prompt and structured summary of facts
    llm = get_llm(temperature=0.0)
    
    system_prompt = (
        "You are an expert financial educator, data quality auditor, and friendly data storyteller. "
        "Your task is to transform verified JSON financial facts into a rich, narrative-driven financial intelligence report "
        "that is deeply engaging, easy to read, and fully detailed across all sections for executive review and RAG chatbot retrieval. "
        "CRITICAL RULES: "
        "1. NARRATIVE STYLE: Write in smooth, natural, professional paragraphs and flowing sentences. "
        "2. STRICT GROUNDING: Never invent, calculate, or estimate numbers. Use ONLY exact figures from the JSON payload. "
        "3. PLAIN-ENGLISH EDUCATION: Explain financial terms (mean, median, IQR, percentiles) immediately in everyday words. "
        "4. FORMATTING: Do NOT use math symbols like multiplication signs (×) or unicode dashes. Write them out in plain text. Avoid markdown hashes (#) or tables."
    )

    human_prompt = f"""
    Please generate an exhaustive, richly detailed Executive Financial Intelligence Report based strictly on these verified JSON facts:

    {json.dumps(facts_data, indent=2)}

    Use clean numbered sections (1, 2, 3, 4, 5, 6) with comprehensive narrative paragraphs and clear inline breakdowns covering:
    1. Executive Summary & Big Picture Overview (In-depth narrative explaining dataset scope, total portfolio value in Indian Rupees, date coverage from January to December 2025, monthly volume and revenue highlights, and overall portfolio health)
    2. Data Quality & Missing Information Audit (Thorough audit narrative detailing exact duplicate row counts, missing cell distributions across columns, invalid dates and amounts, and why clean records matter)
    3. Transaction Statistics & Granular Breakdown (Explain overall statistics with clear plain-English definitions for mean, median, percentiles, and IQR, followed by complete monthly trends, category distributions, payment statuses, channels, and regional activity)
    4. Unusual Activity & Giant Spikes (Explain the IQR outlier method in everyday words, detail upper and lower threshold fences, and describe review candidate transactions naturally)
    5. Actionable Advice for Risk Mitigation & Data Pipelines (Practical recommendations for data hygiene, anomaly monitoring, and risk controls)
    6. Overall Conclusion and Report Limitations (Summarize verified findings and state report boundaries in a polished closing statement)
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)
    raw_content = response.content

    if not raw_content or len(raw_content.strip()) < 50:
        raise ValueError("LLM returned an empty response. Check token limits or input payload size.")

    # 2. Save Markdown Report
    md_path = os.path.join(run_dir, "final_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(raw_content)

    # 3. Compile Professional PDF with reliable line parsing
    pdf_path = os.path.join(run_dir, "final_report.pdf")
    pdf = ExecutivePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for line in raw_content.split('\n'):
        if not line.strip():
            pdf.ln(3)
            continue
        
        line_clean = sanitize_text(line).strip()
        
        if '|' in line_clean or '---' in line_clean:
            continue

        line_clean = line_clean.replace('**', '').replace('__', '').replace('`', '').replace('*', '')

        # Check if it's a main heading
        if line.strip().startswith('#') or (len(line_clean) > 2 and line_clean[0].isdigit() and line_clean[1] == '.' and len(line_clean) < 70 and not line_clean.lower().startswith('INR')):
            pdf.ln(5)
            pdf.set_font('helvetica', 'B', 11)
            pdf.set_text_color(15, 23, 42)
            heading_text = line_clean.replace('#', '').strip()
            pdf.multi_cell(190, 6, heading_text)
            pdf.ln(1.5)
        elif line.strip().startswith('- ') or line.strip().startswith('* ') or line.strip().startswith('• '):
            pdf.set_font('helvetica', '', 9.5)
            pdf.set_text_color(51, 65, 85)
            pdf.set_x(14)
            bullet_text = line_clean.lstrip('-*• ').strip()
            pdf.multi_cell(182, 5, f"-  {bullet_text}")
        else:
            pdf.set_font('helvetica', '', 9.5)
            pdf.set_text_color(51, 65, 85)
            pdf.set_x(10)
            pdf.multi_cell(190, 5, line_clean)

    # Embed charts cleanly on a dedicated analytics page
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Visual Analytics & Chart Artifacts", 0, 1, 'L')
    pdf.ln(4)

    chart_files = [
        ("Amount Distribution (IQR & Spikes)", "amount_distribution.png"),
        ("Transaction Volume Over Time", "volume_over_time.png"),
        ("Missing Values Audit", "missing_values.png"),
        ("Category & Channel Breakdown", "category_channel_breakdown.png")
    ]

    for title, filename in chart_files:
        img_path = os.path.join(run_dir, filename)
        if os.path.exists(img_path):
            pdf.set_font('helvetica', 'B', 10)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(0, 6, title, 0, 1)
            pdf.image(img_path, w=155)
            pdf.ln(6)

    pdf.output(pdf_path)

    return {
        "markdown_report_path": md_path.replace("\\", "/"),
        "pdf_report_path": pdf_path.replace("\\", "/")
    }