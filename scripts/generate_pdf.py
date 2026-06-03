from markdown_pdf import MarkdownPdf, Section

pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(open("PROJECT_REPORT.md", "r", encoding="utf-8").read()))
pdf.meta["title"] = "Axiom Project Report"
pdf.meta["author"] = "Axiom Autonomous AI"
pdf.save("Axiom_Project_Report.pdf")
print("PDF generation complete!")
