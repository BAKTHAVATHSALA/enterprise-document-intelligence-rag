"""Script to generate real PDF test fixtures for Phase 1 testing.

Creates test_documents/ directory containing:
1. contract_abc.pdf (2 pages)
2. contract_xyz.pdf (2 pages)
3. technical_policy.pdf (2 pages)
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_documents")


def build_contract_abc() -> str:
    """Generate contract_abc.pdf test fixture (2 distinct pages)."""
    pdf_path = os.path.join(OUTPUT_DIR, "contract_abc.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=14, leading=18)
    body_style = styles["Normal"]

    # Page 1
    story.append(Paragraph("MASTER SERVICES AGREEMENT", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Document Reference:</b> Contract #123", body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("1. PARTIES", h2_style))
    story.append(
        Paragraph(
            "This Agreement is entered into by and between <b>ABC Corp</b> ('Client') and ServiceProvider LLC. "
            "Primary Contact: John Smith (Email: john@example.com, Phone: +1-555-123-4567, SSN: 123-45-6789).",
            body_style,
        )
    )
    story.append(Spacer(1, 18))
    story.append(Paragraph("2. PAYMENT TERMS", h2_style))
    story.append(
        Paragraph(
            "Client ABC Corp agrees to pay invoice amounts within thirty (30) days of receipt. "
            "Late payments shall accrue interest at 1.5% per month.",
            body_style,
        )
    )

    # Force Page 2
    story.append(Spacer(1, 200))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. TERMINATION CLAUSE", h2_style))
    story.append(
        Paragraph(
            "Either party may terminate Contract #123 by providing at least <b>30 days written notice</b> to the other party. "
            "Upon termination, ABC Corp shall settle all outstanding fees incurred up to the effective termination date.",
            body_style,
        )
    )
    story.append(Spacer(1, 18))
    story.append(Paragraph("4. CONFIDENTIALITY", h2_style))
    story.append(
        Paragraph(
            "Both ABC Corp and ServiceProvider agree to maintain strict confidentiality regarding proprietary technical data and trade secrets.",
            body_style,
        )
    )

    doc.build(story)
    return pdf_path


def build_contract_xyz() -> str:
    """Generate contract_xyz.pdf test fixture (2 distinct pages)."""
    pdf_path = os.path.join(OUTPUT_DIR, "contract_xyz.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=14, leading=18)
    body_style = styles["Normal"]

    # Page 1
    story.append(Paragraph("SOFTWARE LICENSE AGREEMENT", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Document Reference:</b> Contract #456", body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("1. LICENSOR & LICENSEE", h2_style))
    story.append(
        Paragraph(
            "This Agreement is executed by <b>XYZ Technologies</b> ('Licensor') and Enterprise Global Inc. "
            "Key Contact: Alice Johnson (Email: alice@xyztech.org, Phone: +1-555-987-6543).",
            body_style,
        )
    )
    story.append(Spacer(1, 18))
    story.append(Paragraph("2. SCOPE OF LICENSE", h2_style))
    story.append(
        Paragraph(
            "XYZ Technologies grants Enterprise Global Inc. a non-exclusive license to deploy software modules across corporate servers.",
            body_style,
        )
    )

    # Force Page 2
    story.append(Spacer(1, 200))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. CANCELLATION & TERMINATION", h2_style))
    story.append(
        Paragraph(
            "Contract #456 may be terminated by either party with <b>60 days written notice</b>. "
            "XYZ Technologies reserves the right to audit software compliance annually.",
            body_style,
        )
    )

    doc.build(story)
    return pdf_path


def build_technical_policy() -> str:
    """Generate technical_policy.pdf test fixture (2 distinct pages)."""
    pdf_path = os.path.join(OUTPUT_DIR, "technical_policy.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=14, leading=18)
    body_style = styles["Normal"]

    # Page 1
    story.append(Paragraph("ENTERPRISE CLOUD SECURITY POLICY", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Policy ID:</b> POL-2026-SEC", body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("1. ACCESS CONTROL STANDARDS", h2_style))
    story.append(
        Paragraph(
            "All cloud service deployments must enforce multi-factor authentication (MFA) and least-privilege role access.",
            body_style,
        )
    )
    story.append(Spacer(1, 18))
    story.append(Paragraph("2. DATA CLASSIFICATION MATRIX", h2_style))

    table_data = [
        ["Classification", "Description", "Encryption Required"],
        ["Public", "Marketing content", "Optional"],
        ["Internal", "Employee guidelines", "TLS in transit"],
        ["Confidential", "Financials & Contracts", "AES-256 at rest"],
    ]
    t = Table(table_data, colWidths=[120, 180, 140])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(t)

    # Force Page 2
    story.append(Spacer(1, 200))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. INCIDENT RESPONSE PLAN", h2_style))
    story.append(
        Paragraph(
            "Security incidents must be reported to security@enterprise.com within 2 hours of detection. "
            "For urgent escalation call +1-800-555-0199.",
            body_style,
        )
    )

    doc.build(story)
    return pdf_path


def generate_all_pdfs() -> None:
    """Generate all test PDF fixtures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p1 = build_contract_abc()
    p2 = build_contract_xyz()
    p3 = build_technical_policy()
    print(f"Generated multi-page test PDF fixtures:\n - {p1}\n - {p2}\n - {p3}")


if __name__ == "__main__":
    generate_all_pdfs()
