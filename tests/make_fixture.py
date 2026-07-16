"""Generate a structurally-identical FIXTURE workbook for testing.

This is NOT the real dataset. It exists so the parser and model can be exercised
end-to-end without the real workbook present. It deliberately reproduces the real
file's quirks:
  - a banner row and an "End Date" row above the real header (header lands on row 3)
  - 22 columns
  - ';'-separated Fields / Report Prompts
  - exact-duplicate report definitions (near-duplicate families)
  - null Last Run / Last Updated dates
  - a Description column with only a handful of distinct values

Usage:  python tests/make_fixture.py [n_rows] [out_path]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

RNG = random.Random(20240517)

CATEGORIES = [
    "Human Resources", "Payroll", "Talent", "Recruiting", "Time Tracking",
    "Absence", "Benefits", "Compensation", "Learning", "Workforce Planning",
    "Financials", "Expenses", "Procurement", "Projects", "Security",
    "Integrations", "Audit", "Diversity", "Onboarding", "Offboarding", "Reporting",
]

DATA_SOURCES = [
    "Workers for HCM", "All Workers", "All Active Employees", "Terminated Workers",
    "Worker History", "Job Profiles", "Positions", "Supervisory Organizations",
    "Payroll Results", "Payroll Input", "Compensation Basis", "Time Blocks",
    "Absence Balances", "Benefit Elections", "Job Requisitions", "Candidates",
    "Learning Enrollments", "Cost Centers", "Companies", "Journal Lines",
    "Expense Reports", "Purchase Orders", "Suppliers", "Projects and Tasks",
    "Security Groups", "Business Processes", "Organizations", "Locations",
]

REPORT_TYPES = [
    "Advanced", "Simple", "Matrix", "Composite", "Search", "Trending",
    "nBox", "Transposed",
]

TAGS = [
    "HR Ops", "Monthly", "Quarterly", "Executive", "Compliance", "Audit",
    "Finance", "Ad Hoc", "Dashboard", "Reconciliation", "Headcount", "Attrition",
    "Payroll Ops", "Benefits Ops", "Talent Review", "Recruiting Ops",
    "Time Ops", "Data Quality", "Integration", "Legacy", "Restricted", "Shared Services",
]

# Topic templates: (title patterns, characteristic fields, prompts)
TOPICS = [
    (
        ["Worker Transfer History", "Employee Transfers by Organization",
         "Worker Job Change History", "Internal Movement Detail",
         "Transfer and Promotion History"],
        ["Employee ID", "Worker", "Legal Name", "Transfer Date", "Previous Supervisory Organization",
         "New Supervisory Organization", "Previous Job Profile", "New Job Profile",
         "Business Process Type", "Effective Date", "Reason", "Manager"],
        ["Effective Date Range", "Supervisory Organization", "Business Process Type"],
    ),
    (
        ["Monthly Turnover Summary", "Turnover Rate by Organization",
         "Attrition Summary by Month", "Voluntary Turnover Trend",
         "Headcount and Turnover Rollup"],
        ["Supervisory Organization", "Termination Date", "Termination Reason",
         "Voluntary Termination", "Headcount Beginning", "Headcount Ending",
         "Turnover Rate", "Average Headcount", "Month", "Company"],
        ["Period", "Company", "Supervisory Organization", "Include Subordinate Organizations"],
    ),
    (
        ["Active Headcount by Company", "Current Worker Snapshot",
         "Active Employee Headcount Detail", "Headcount by Company and Location",
         "Worker Roster Snapshot"],
        ["Employee ID", "Worker", "Company", "Location", "Supervisory Organization",
         "Job Profile", "Worker Type", "Employee Status", "Hire Date", "FTE",
         "Manager", "Cost Center"],
        ["As of Date", "Company", "Location", "Include Contingent Workers"],
    ),
    (
        ["New Hires This Quarter", "New Hire Report with Start Dates",
         "Recent Hires by Supervisory Organization", "Onboarding Status of New Hires",
         "Quarterly Hiring Detail"],
        ["Employee ID", "Worker", "Hire Date", "Original Hire Date", "Start Date",
         "Job Profile", "Supervisory Organization", "Location", "Recruiter",
         "Onboarding Status", "Worker Type", "Manager"],
        ["Hire Date Range", "Supervisory Organization", "Worker Type"],
    ),
    (
        ["Terminated Workers by Supervisory Organization", "Termination Detail Report",
         "Involuntary Termination Audit", "Exiting Workers with Last Day of Work",
         "Termination Reason Analysis"],
        ["Employee ID", "Worker", "Termination Date", "Last Day of Work",
         "Termination Reason", "Termination Category", "Voluntary Termination",
         "Supervisory Organization", "Manager", "Job Profile", "Company", "Eligible for Rehire"],
        ["Termination Date Range", "Supervisory Organization", "Termination Reason"],
    ),
    (
        ["Payroll Register Detail", "Gross to Net Payroll Summary",
         "Payroll Results by Pay Group", "Earnings and Deductions Detail",
         "Payroll Reconciliation Report"],
        ["Employee ID", "Worker", "Pay Group", "Pay Period", "Gross Pay", "Net Pay",
         "Earnings Code", "Deduction Code", "Tax Amount", "Payment Date", "Company"],
        ["Pay Period", "Pay Group", "Company"],
    ),
    (
        ["Compensation Change History", "Merit Increase Detail",
         "Salary Adjustment Audit", "Compensation Basis by Worker",
         "Base Pay Change Report"],
        ["Employee ID", "Worker", "Effective Date", "Compensation Plan", "Previous Base Pay",
         "New Base Pay", "Percent Change", "Currency", "Compensation Grade",
         "Job Profile", "Manager"],
        ["Effective Date Range", "Compensation Plan", "Supervisory Organization"],
    ),
    (
        ["Absence Balance by Worker", "Time Off Taken Detail",
         "Leave of Absence Report", "PTO Accrual Balances",
         "Absence Utilization Summary"],
        ["Employee ID", "Worker", "Absence Type", "Balance", "Units", "Accrued",
         "Taken", "Leave Start Date", "Leave End Date", "Supervisory Organization"],
        ["As of Date", "Absence Type", "Supervisory Organization"],
    ),
    (
        ["Benefit Election Detail", "Open Enrollment Status",
         "Dependent Coverage Report", "Benefit Plan Participation",
         "Benefits Eligibility Audit"],
        ["Employee ID", "Worker", "Benefit Plan", "Coverage Level", "Election Date",
         "Dependent Name", "Relationship", "Employee Cost", "Employer Cost", "Plan Year"],
        ["Plan Year", "Benefit Plan", "Company"],
    ),
    (
        ["Job Requisition Pipeline", "Open Requisitions by Organization",
         "Candidate Pipeline Detail", "Time to Fill Analysis",
         "Recruiting Funnel Summary"],
        ["Job Requisition", "Requisition Status", "Hiring Manager", "Recruiter",
         "Candidate", "Candidate Stage", "Application Date", "Days Open",
         "Supervisory Organization", "Location", "Job Profile"],
        ["Requisition Status", "Recruiter", "Date Range"],
    ),
    (
        ["Learning Enrollment Status", "Required Training Completion",
         "Course Completion by Organization", "Learning Assignment Audit",
         "Compliance Training Detail"],
        ["Employee ID", "Worker", "Course", "Enrollment Date", "Completion Date",
         "Completion Status", "Required", "Due Date", "Supervisory Organization"],
        ["Course", "Completion Status", "Due Date Range"],
    ),
    (
        ["Journal Line Detail", "General Ledger Reconciliation",
         "Cost Center Spend Summary", "Accounting Journal Audit",
         "Ledger Account Activity"],
        ["Journal Number", "Journal Line", "Ledger Account", "Cost Center", "Company",
         "Debit Amount", "Credit Amount", "Accounting Date", "Currency", "Journal Source"],
        ["Accounting Period", "Company", "Ledger Account"],
    ),
    (
        ["Expense Report Detail", "Expense Reimbursement Audit",
         "Out of Policy Expense Report", "Corporate Card Transactions",
         "Expense Spend by Cost Center"],
        ["Expense Report Number", "Worker", "Expense Item", "Amount", "Currency",
         "Expense Date", "Approval Status", "Cost Center", "Company", "Policy Exception"],
        ["Expense Date Range", "Approval Status", "Cost Center"],
    ),
    (
        ["Purchase Order Detail", "Supplier Spend Analysis",
         "Open Purchase Orders", "Procurement Cycle Time",
         "Supplier Invoice Reconciliation"],
        ["Purchase Order", "Supplier", "PO Line", "Item", "Quantity", "Unit Cost",
         "Total Amount", "PO Status", "Order Date", "Cost Center", "Company"],
        ["Order Date Range", "Supplier", "PO Status"],
    ),
    (
        ["Security Group Membership", "User Access Audit",
         "Domain Security Policy Report", "Role Assignment Detail",
         "Segregation of Duties Review"],
        ["Security Group", "Worker", "Role", "Assignment Date", "Domain",
         "Permission Type", "Supervisory Organization", "Last Login"],
        ["Security Group", "Domain", "Worker"],
    ),
    (
        ["Diversity Representation Summary", "Gender Representation by Level",
         "Ethnicity Distribution Report", "Pay Equity Analysis",
         "Inclusion Metrics Dashboard"],
        ["Supervisory Organization", "Gender", "Ethnicity", "Job Level",
         "Headcount", "Percent of Total", "Average Base Pay", "Company", "Location"],
        ["As of Date", "Company", "Job Level"],
    ),
    (
        ["Worker Time Block Detail", "Overtime Hours by Organization",
         "Time Entry Audit", "Hours Worked Summary",
         "Time Tracking Exception Report"],
        ["Employee ID", "Worker", "Time Block Date", "Hours", "Time Entry Code",
         "Overtime Hours", "Approval Status", "Supervisory Organization", "Cost Center"],
        ["Date Range", "Time Entry Code", "Supervisory Organization"],
    ),
    (
        ["Position Management Detail", "Vacant Positions by Organization",
         "Position Budget vs Actual", "Headcount Plan Detail",
         "Staffing Model Audit"],
        ["Position ID", "Position", "Job Profile", "Supervisory Organization",
         "Position Status", "Filled", "Vacant", "Availability Date", "FTE", "Company"],
        ["As of Date", "Supervisory Organization", "Position Status"],
    ),
    (
        ["Performance Review Status", "Talent Review Calibration",
         "Goal Completion by Organization", "Performance Rating Distribution",
         "Succession Plan Readiness"],
        ["Employee ID", "Worker", "Review Period", "Review Status", "Performance Rating",
         "Potential Rating", "Goal Count", "Goals Completed", "Manager",
         "Supervisory Organization"],
        ["Review Period", "Supervisory Organization", "Review Status"],
    ),
    (
        ["Integration Event Audit", "Failed Integration Runs",
         "EIB Load Results", "Integration System Activity",
         "Data Quality Exception Report"],
        ["Integration System", "Event Status", "Run Date", "Records Processed",
         "Records Failed", "Error Message", "Initiated By", "Duration"],
        ["Run Date Range", "Integration System", "Event Status"],
    ),
]

DESCRIPTIONS = [
    "Custom report created for business reporting needs.",
    "Standard operational report.",
    "Report used for periodic review.",
    "Ad hoc report built on request.",
    "Report maintained by the reporting team.",
    "Legacy report retained for reference.",
    "Report supporting compliance requirements.",
]

QUALIFIERS = [
    "", "", "", " - Detail", " - Summary", " (Copy)", " v2", " - Monthly",
    " - Quarterly", " - Archived", " - EMEA", " - APAC", " - NA", " (Legacy)",
    " - Restricted", " - Draft",
]

# The real workbook has ~3280 distinct titles across 4000 rows. Title variety has
# to come from somewhere, so titles are composed from a wide combinatorial space
# (stem x dimension x qualifier) rather than a short fixed list. Under-generating
# distinct titles makes the corpus far more redundant than the real estate and
# spreads the posterior across near-identical competitors, which would make the
# decision rule look miscalibrated when it isn't.
DIMENSIONS = [
    "", "", " by Supervisory Organization", " by Company", " by Location",
    " by Cost Center", " by Manager", " by Job Profile", " by Worker Type",
    " by Region", " by Business Unit", " by Pay Group", " by Month",
    " by Quarter", " for Active Workers", " for Contingent Workers",
]

COLUMNS = [
    "Custom Report", "Description", "Category", "Data Source", "Report Type",
    "Fields", "Report Prompts", "Report Tag(s)", "Number of Times",
    "Last Run Date", "Last Updated Date", "Shared", "Owner", "Enabled",
    "Web Service Enabled", "Temporary Report", "Report Definition ID",
    "Optimized for Performance", "Filter Count", "Sort Count",
    "Column Count", "Data Source Filter",
]


def _make_row(topic_idx: int) -> dict:
    titles, fields_pool, prompts_pool = TOPICS[topic_idx]
    title = RNG.choice(titles) + RNG.choice(DIMENSIONS) + RNG.choice(QUALIFIERS)

    n_fields = RNG.randint(5, min(12, len(fields_pool)))
    fields = RNG.sample(fields_pool, n_fields)

    if RNG.random() < 0.82:
        n_prompts = RNG.randint(1, len(prompts_pool))
        prompts = "; ".join(RNG.sample(prompts_pool, n_prompts))
    else:
        prompts = None

    last_run = (
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=RNG.randint(0, 540))
        if RNG.random() < 0.75
        else None
    )
    last_updated = (
        pd.Timestamp("2023-01-01") + pd.Timedelta(days=RNG.randint(0, 900))
        if RNG.random() < 0.85
        else None
    )

    return {
        "Custom Report": title,
        "Description": RNG.choice(DESCRIPTIONS),
        "Category": RNG.choice(CATEGORIES),
        "Data Source": RNG.choice(DATA_SOURCES),
        "Report Type": RNG.choice(REPORT_TYPES),
        "Fields": "; ".join(fields),
        "Report Prompts": prompts,
        "Report Tag(s)": RNG.choice(TAGS) if RNG.random() < 0.78 else None,
        "Number of Times": RNG.randint(0, 2400),
        "Last Run Date": last_run,
        "Last Updated Date": last_updated,
        "Shared": RNG.choice(["Yes", "No"]),
        "Owner": f"User {RNG.randint(1000, 9999)}",
        "Enabled": RNG.choice(["Yes", "No"]),
        "Web Service Enabled": RNG.choice(["Yes", "No"]),
        "Temporary Report": RNG.choice(["Yes", "No"]),
        "Report Definition ID": f"RPT-{RNG.randint(100000, 999999)}",
        "Optimized for Performance": RNG.choice(["Yes", "No"]),
        "Filter Count": RNG.randint(0, 8),
        "Sort Count": RNG.randint(0, 4),
        "Column Count": n_fields,
        "Data Source Filter": RNG.choice(["", "Active Only", "Current Period", "All"]),
    }


def build(n_rows: int = 4000) -> pd.DataFrame:
    rows: list[dict] = []
    # ~80% fresh definitions; the rest are exact duplicates of earlier rows so the
    # family-collapse path has something real to collapse.
    n_unique = int(n_rows * 0.8)
    for i in range(n_unique):
        rows.append(_make_row(i % len(TOPICS)))
    for _ in range(n_rows - n_unique):
        source = dict(RNG.choice(rows))
        source["Number of Times"] = RNG.randint(0, 2400)
        source["Owner"] = f"User {RNG.randint(1000, 9999)}"
        rows.append(source)
    RNG.shuffle(rows)
    return pd.DataFrame(rows, columns=COLUMNS)


def write(path: Path, n_rows: int = 4000) -> None:
    frame = build(n_rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Two junk rows above the header, exactly like the real workbook.
        banner = pd.DataFrame([["All Custom Reports — Synthetic Estate"] + [None] * 21])
        end_date = pd.DataFrame([["End Date", "2025-06-30"] + [None] * 20])
        banner.to_excel(writer, index=False, header=False, startrow=0, sheet_name="Sheet1")
        end_date.to_excel(writer, index=False, header=False, startrow=1, sheet_name="Sheet1")
        frame.to_excel(writer, index=False, header=True, startrow=2, sheet_name="Sheet1")

    print(f"Wrote fixture: {path} ({len(frame)} rows x {len(frame.columns)} cols, header on row 3)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "fixtures" / "fixture_reports.xlsx"
    write(out, n)
