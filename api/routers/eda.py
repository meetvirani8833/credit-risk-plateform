from fastapi import APIRouter

router = APIRouter(prefix="/eda", tags=["eda"])

# Mirrors notebooks/eda.py section 5 (Business Insights). Kept as a small
# static structure here rather than recomputing over the full dataset on
# every request, the insights don't change, and the deployed backend
# shouldn't need the full 307K-row application_train loaded into memory
# just to answer "what are the EDA insights."
INSIGHTS = [
    {
        "title": "External source scores are the strongest single predictors",
        "chart": "03_ext_source_vs_target.png",
        "summary": (
            "All three EXT_SOURCE scores are negatively correlated with default "
            "and show clearly separated distributions between repaid and "
            "not-repaid clients, the strongest individual signals in the dataset."
        ),
    },
    {
        "title": "Younger and less employment-stable applicants default more",
        "chart": "04_default_by_age.png",
        "summary": (
            "Non-repayment falls steadily with age: the 20-30 age band defaults "
            "at roughly 2.3x the rate of the 60-70 band."
        ),
    },
    {
        "title": "Income type is a strong differentiator, especially at the extremes",
        "chart": "05_default_by_income_type.png",
        "summary": (
            "Working applicants default almost twice as often as Pensioners "
            "(9.6% vs 5.4%). Maternity leave and Unemployed segments default "
            "far more (40% and 36%), though on small sample sizes."
        ),
    },
    {
        "title": "Debt burden relative to income separates the two classes",
        "chart": "06_ratios_vs_target.png",
        "summary": (
            "Clients who default carry a slightly higher annuity-to-income "
            "burden, motivating the engineered CREDIT_INCOME_RATIO and "
            "ANNUITY_INCOME_RATIO features."
        ),
    },
    {
        "title": "Prior credit bureau history is an independent risk signal",
        "chart": "07_bureau_history_vs_target.png",
        "summary": (
            "Applicants with no prior Credit Bureau history default more "
            "(10.1%) than those with history (7.7%), justifying the bureau.csv "
            "feature aggregation used in the model."
        ),
    },
]

DATA_QUALITY = {
    "class_imbalance_pct": 8.07,
    "columns_with_missing": 67,
    "total_columns": 122,
    "days_employed_sentinel_rows": 55374,
    "notes": [
        "8.07% positive rate, accuracy is not a meaningful metric here.",
        "40+ building/apartment metadata columns are over 50% missing and were dropped.",
        "DAYS_EMPLOYED uses a 365243 sentinel for 'not employed' (18% of rows), "
        "replaced with NaN plus an IS_EMPLOYED flag rather than used as a raw number.",
        "4 rows with CODE_GENDER='XNA' and 3 extreme AMT_INCOME_TOTAL outliers "
        "(up to 117,000,000) were found and handled explicitly.",
        "No duplicate SK_ID_CURR or fully duplicate rows.",
    ],
}


@router.get("/insights")
def get_insights():
    return {"insights": INSIGHTS, "data_quality": DATA_QUALITY}
