# This agent creates final PDF and Markdown reports with all analysis results
import os
from fpdf import FPDF
from matplotlib import font_manager

# Create folder for saved reports
OUTPUT_DIR = "outputs/reports/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Custom PDF class to format the report nicely
class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self._font_family = self._register_unicode_fonts()

    def _register_unicode_fonts(self):
        try:
            regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
            bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
            italic = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", style="italic"))
            bold_italic = font_manager.findfont(
                font_manager.FontProperties(family="DejaVu Sans", weight="bold", style="italic")
            )

            self.add_font("DejaVu", "", regular)
            self.add_font("DejaVu", "B", bold)
            self.add_font("DejaVu", "I", italic)
            self.add_font("DejaVu", "BI", bold_italic)
            return "DejaVu"
        except Exception:
            return "Arial"

    # Add title at top of each page
    def header(self):
        self.set_font(self._font_family, "B", 14)
        self.cell(0, 10, "Automated Data Analysis Report", ln=True, align="C")
        self.ln(4)

    # Add page number at bottom of each page
    def footer(self):
        self.set_y(-15)
        self.set_font(self._font_family, "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # Add a text section with title and content
    def add_section(self, title: str, body: str):
        self.set_font(self._font_family, "B", 12)
        self.cell(0, 8, title, ln=True)
        self.set_font(self._font_family, "", 10)
        self.multi_cell(0, 6, body)
        self.ln(4)

    # Add an image to the report
    def add_image(self, img_path: str, title: str):
        if os.path.exists(img_path):
            self.set_font(self._font_family, "B", 10)
            self.cell(0, 6, title, ln=True)
            self.image(img_path, w=170)
            self.ln(4)


# Main function that combines all analysis into one PDF and Markdown report
def run_report_agent(
    clean_report: str,
    eda_results: dict,
    viz_results: dict,
    insights: str,
    filename: str = "report.pdf"
) -> str:
    # Create a new PDF document
    pdf = ReportPDF()
    pdf.add_page()

    # Add all sections to the PDF report
    pdf.add_section("1. Data Cleaning Summary", clean_report)

    pdf.add_section("2. Exploratory Data Analysis", eda_results.get("eda_summary", ""))

    pdf.add_section("3. Visualizations", viz_results.get("viz_summary", ""))
    # Add up to 6 charts to the report
    for chart_path in viz_results.get("chart_paths", [])[:6]:
        chart_name = os.path.basename(chart_path).replace("_", " ").replace(".png", "")
        pdf.add_image(chart_path, chart_name)

    pdf.add_section("4. Business Insights", insights)

    # Save PDF file
    report_path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(report_path)

    # Also save as Markdown file for easy reading
    md_path = report_path.replace(".pdf", ".md")
    with open(md_path, "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## 1. Data Cleaning\n" + clean_report + "\n\n")
        f.write("## 2. EDA Findings\n" + eda_results.get("eda_summary", "") + "\n\n")
        f.write("## 3. Business Insights\n" + insights + "\n\n")

    return report_path
