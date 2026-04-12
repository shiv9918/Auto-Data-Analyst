import os
from fpdf import FPDF

OUTPUT_DIR = "outputs/reports/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Automated Data Analysis Report", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_section(self, title: str, body: str):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, ln=True)
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, body)
        self.ln(4)

    def add_image(self, img_path: str, title: str):
        if os.path.exists(img_path):
            self.set_font("Arial", "B", 10)
            self.cell(0, 6, title, ln=True)
            self.image(img_path, w=170)
            self.ln(4)

def run_report_agent(
    clean_report: str,
    eda_results: dict,
    viz_results: dict,
    insights: str,
    filename: str = "report.pdf"
) -> str:
    pdf = ReportPDF()
    pdf.add_page()

    # Section 1: Cleaning
    pdf.add_section("1. Data Cleaning Summary", clean_report)

    # Section 2: EDA
    pdf.add_section("2. Exploratory Data Analysis", eda_results.get("eda_summary", ""))

    # Section 3: Visualizations
    pdf.add_section("3. Visualizations", viz_results.get("viz_summary", ""))
    for chart_path in viz_results.get("chart_paths", [])[:6]:  # limit to 6 charts
        chart_name = os.path.basename(chart_path).replace("_", " ").replace(".png", "")
        pdf.add_image(chart_path, chart_name)

    # Section 4: Insights
    pdf.add_section("4. Business Insights", insights)

    # Save PDF
    report_path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(report_path)

    # Also save markdown
    md_path = report_path.replace(".pdf", ".md")
    with open(md_path, "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## 1. Data Cleaning\n" + clean_report + "\n\n")
        f.write("## 2. EDA Findings\n" + eda_results.get("eda_summary", "") + "\n\n")
        f.write("## 3. Business Insights\n" + insights + "\n\n")

    return report_path